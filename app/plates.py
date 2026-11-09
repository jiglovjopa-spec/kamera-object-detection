"""Накопление чтений номера по нескольким кадрам и журнал.

Один кадр может дать ошибку, поэтому номер считается прочитанным, только когда
в нескольких кадрах подряд получился один и тот же текст.
"""
from __future__ import annotations

import csv
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CONFIRM_VOTES = 3  # сколько одинаковых чтений нужно
CONFIRM_SHARE = 0.5  # доля этого текста среди всех чтений номера
MIN_WEAKEST = 0.45  # чтение с настолько сомнительным символом не считаем
TRACK_TIMEOUT = 2.0  # через сколько секунд забываем номер, пропавший из кадра
REPEAT_SECONDS = 300  # один и тот же номер пишем в журнал не чаще раза в 5 минут
MIN_FREE_BYTES = 1024 ** 3  # при свободном месте меньше 1 ГБ снимки не сохраняем


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    box: tuple
    seen: float
    votes: Counter = field(default_factory=Counter)
    logged: bool = False

    def best(self) -> tuple[str, bool]:
        if not self.votes:
            return "", False
        text, n = self.votes.most_common(1)[0]
        return text, n >= CONFIRM_VOTES and n / sum(self.votes.values()) >= CONFIRM_SHARE


class PlateTracker:
    def __init__(self):
        self.tracks: list[Track] = []

    def update(self, boxes: list, now: float | None = None) -> list[tuple]:
        """Принимает рамки номеров с кадра, возвращает [(рамка, текст, подтверждён, трек)]."""
        now = time.monotonic() if now is None else now
        out = []
        for b in boxes:
            rect = (b.x1, b.y1, b.x2, b.y2)
            track = max(self.tracks, key=lambda t: _iou(t.box, rect), default=None)
            if track is None or _iou(track.box, rect) < 0.1:
                track = Track(rect, now)
                self.tracks.append(track)
            track.box, track.seen = rect, now
            if b.reading and b.reading.weakest >= MIN_WEAKEST:
                track.votes[b.reading.text] += 1
            text, confirmed = track.best()
            out.append((b, text or "?", confirmed, track))
        self.tracks = [t for t in self.tracks if now - t.seen < TRACK_TIMEOUT]
        return out


class PlateJournal:
    def __init__(self, folder: Path):
        self.folder = folder
        self.photos = folder / "снимки"
        self.csv_path = folder / "журнал_номеров.csv"
        self.photos.mkdir(parents=True, exist_ok=True)
        self.last: dict[str, float] = {}
        self.pending: list[list[str]] = []  # записи, которые пока не удалось дописать в файл
        self.last_retry = 0.0

    def add(self, text: str, frame_bgr: np.ndarray, rect: tuple, now: float | None = None) -> bool:
        """Записать номер. Возвращает True, если запись новая."""
        now = time.monotonic() if now is None else now
        if now - self.last.get(text, -1e9) < REPEAT_SECONDS:
            return False
        self.last[text] = now
        stamp = datetime.now()
        photo_name = self._save_photo(text, stamp, frame_bgr, rect)
        self.pending.append([f"{stamp:%d.%m.%Y}", f"{stamp:%H:%M:%S}", text, photo_name])
        self._write_pending()
        return True

    def retry(self) -> None:
        """Файл журнала мог быть открыт в Excel: раз в 5 секунд пробуем дописать отложенные записи."""
        if self.pending and time.monotonic() - self.last_retry >= 5:
            self._write_pending()

    def _save_photo(self, text: str, stamp: datetime, frame_bgr: np.ndarray, rect: tuple) -> str:
        try:
            if shutil.disk_usage(self.folder).free < MIN_FREE_BYTES:
                self.photo_warning = "Мало места на диске: снимки номеров не сохраняются"
                return ""
            picture = frame_bgr.copy()
            cv2.rectangle(picture, rect[:2], rect[2:], (0, 220, 255), 2)
            name = f"{stamp:%Y-%m-%d_%H-%M-%S}_{text}.jpg"
            # imwrite не умеет русские пути на Windows, поэтому кодируем и пишем сами
            ok, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise OSError("не удалось сжать снимок")
            (self.photos / name).write_bytes(encoded.tobytes())
            self.photo_warning = ""
            return name
        except OSError as error:
            self.photo_warning = f"Снимок номера не сохранён: {error}"
            return ""

    def _write_pending(self) -> None:
        self.last_retry = time.monotonic()
        new_file = not self.csv_path.exists()
        try:
            # utf-8-sig и точка с запятой — чтобы файл сразу открылся в русском Excel
            with self.csv_path.open("a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                if new_file:
                    writer.writerow(["Дата", "Время", "Номер", "Снимок"])
                writer.writerows(self.pending)
            self.pending.clear()
            self.csv_warning = ""
        except OSError:
            self.csv_warning = f"Журнал открыт в другой программе (Excel?). Закройте его: {len(self.pending)} зап. допишутся сами"

    photo_warning = ""
    csv_warning = ""

    @property
    def warning(self) -> str:
        return "\n".join(w for w in (self.photo_warning, self.csv_warning) if w)
