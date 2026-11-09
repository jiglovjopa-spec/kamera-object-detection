"""Окно программы: картинка с камеры, рамки и подписи, журнал номеров."""
from __future__ import annotations

import argparse
import os
import threading
import time
import tkinter as tk
import traceback
from collections import Counter
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from detectors import PlateDetector, SceneDetector
from plates import PlateJournal, PlateTracker

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "результаты"
COLORS = {"object": "#4FA3FF", "veg": "#41D66B", "plate": "#FFC233", "plate_unsure": "#C9A24A"}
RESULT_LIFETIME = 1.0  # сколько секунд держать рамку, если новых результатов нет
CAMERA_HELP = ("Камера недоступна.\n\nПроверьте: не занята ли она другой программой (Zoom, Telemost и т.п.),\n"
               "включена ли шторка/кнопка камеры на ноутбуке и разрешён ли доступ:\n"
               "Параметры Windows → Конфиденциальность → Камера.\n\nПовторная попытка каждые 3 секунды…")


def load_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "tahoma.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Camera(threading.Thread):
    """Держит самый свежий кадр. source — номер камеры или путь к фото/видео (для проверки программы)."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source, self.frame, self.stamp, self.failed = source, None, 0.0, False
        self.running = True
        self.shots = 0

    def _open(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        return None

    def run(self):
        if not isinstance(self.source, int):
            self._run_file()
            return
        while self.running:
            cap = self._open()
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                self.failed = True
                time.sleep(3)
                continue
            self.failed = False
            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.failed = True
                    break
                self.frame, self.stamp = frame, time.monotonic()
                self.shots += 1
            cap.release()

    def _run_file(self):
        path = Path(self.source)
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            while self.running:
                self.frame, self.stamp = image, time.monotonic()
                self.shots += 1
                time.sleep(0.04)
            return
        while self.running:
            cap = cv2.VideoCapture(str(path))
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame, self.stamp = frame, time.monotonic()
                self.shots += 1
                time.sleep(0.03)
            cap.release()


class Worker(threading.Thread):
    """Считает одну модель на самом свежем кадре, сколько успевает. Ошибки показывает в окне, а не прячет."""

    def __init__(self, name: str, make, camera: Camera):
        super().__init__(daemon=True, name=name)
        self.make, self.camera = make, camera
        self.boxes, self.stamp, self.rate, self.frame = [], 0.0, 0.0, None
        self.error, self.ready, self.on_gpu, self.enabled = "", False, False, True
        self.running = True
        self.step = None

    def run(self):
        try:
            self.step, self.on_gpu = self.make()
        except Exception:
            self.error = f"Не загрузилась модель «{self.name}». Подробности в окне консоли."
            traceback.print_exc()
            return
        self.ready = True
        last = -1
        while self.running:
            frame, stamp = self.camera.frame, self.camera.stamp
            if frame is None or stamp == last or not self.enabled:
                time.sleep(0.005)
                continue
            last = stamp
            started = time.monotonic()
            try:
                self.boxes, self.frame = self.step(frame), frame
                self.stamp, self.error = time.monotonic(), ""
            except Exception:
                self.error = f"Сбой в распознавании «{self.name}». Подробности в окне консоли."
                traceback.print_exc()
                time.sleep(1)
                continue
            dt = max(1e-3, time.monotonic() - started)
            self.rate = 0.8 * self.rate + 0.2 / dt if self.rate else 1 / dt


class App:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        root.title("Камера: предметы, номера, овощи")
        root.geometry("1280x700")
        root.minsize(900, 500)
        self.camera = Camera(args.source if args.source else args.camera)
        self.camera.start()

        self.show_objects, self.show_plates, self.show_veg = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.font, self.small = load_font(18), load_font(14)
        self.scene = None
        self.scene_worker = Worker("предметы и овощи", self._make_scene, self.camera)
        self.plate_worker = Worker("номера", self._make_plates, self.camera)
        self.scene_worker.start()
        self.plate_worker.start()
        self.tracker = PlateTracker()
        self.journal = PlateJournal(RESULTS)
        self.last_plate_stamp = 0.0
        self.plate_view: list[tuple] = []
        self.ui_rate, self.ui_last = 0.0, time.monotonic()
        self.photo = None
        self.started = time.monotonic()
        self.last_canvas = self.veg_canvas = self.plate_canvas = None
        self.seen = Counter()

        self.video = tk.Label(root, bg="#111", fg="#eee", font=("Segoe UI", 13), text="Запуск камеры и загрузка моделей…", justify="left")
        self.video.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(root, padding=10, width=270)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="Что показывать", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        for text, var in (("Предметы (синие рамки)", self.show_objects), ("Овощи (зелёные рамки)", self.show_veg),
                          ("Автомобильные номера (жёлтые)", self.show_plates)):
            ttk.Checkbutton(side, text=text, variable=var).pack(anchor="w", pady=2)
        ttk.Label(side, text="Прочитанные номера", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(16, 2))
        self.plate_list = tk.Listbox(side, height=14, font=("Consolas", 12))
        self.plate_list.pack(fill="both", expand=True)
        ttk.Button(side, text="Открыть журнал и снимки", command=lambda: os.startfile(RESULTS)).pack(fill="x", pady=(8, 2))
        self.status = ttk.Label(side, text="", wraplength=250, justify="left")
        self.status.pack(fill="x", pady=(10, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def _make_scene(self):
        scene = SceneDetector(float(self.args.objects_threshold))
        self.scene = scene

        def step(frame):
            return scene(frame, self.want_objects, self.want_veg)
        return step, scene.on_gpu

    want_objects = True
    want_veg = True

    def _make_plates(self):
        detector = PlateDetector()
        return detector, detector.net.model.get_providers()[0] == "CUDAExecutionProvider"

    def _active(self, worker) -> list:
        return worker.boxes if time.monotonic() - worker.stamp < RESULT_LIFETIME else []

    def _update_plates(self):
        worker = self.plate_worker
        if worker.stamp != self.last_plate_stamp:
            self.last_plate_stamp = worker.stamp
            self.plate_view = self.tracker.update(worker.boxes)
            for box, text, confirmed, track in self.plate_view:
                if confirmed and not track.logged and worker.frame is not None:
                    track.logged = True
                    if self.journal.add(text, worker.frame, (box.x1, box.y1, box.x2, box.y2)):
                        self.plate_list.insert(0, f"{time.strftime('%H:%M:%S')}  {text}")
        if time.monotonic() - worker.stamp >= RESULT_LIFETIME:
            self.plate_view = []

    def _draw(self, frame, width: int, height: int) -> Image.Image:
        fh, fw = frame.shape[:2]
        scale = min(width / fw, height / fh)
        shown = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        def label(b, text, color):
            x1, y1, x2, y2 = (int(v * scale) for v in (b.x1, b.y1, b.x2, b.y2))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=self.font)
            tw, th = right - left, bottom - top
            ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1
            draw.rectangle((x1, ty, x1 + tw + 8, ty + th + 8), fill=color)
            draw.text((x1 + 4, ty + 2), text, fill="#101010", font=self.font)

        scene = self._active(self.scene_worker)
        if self.show_objects.get():
            for b in scene:
                if b.kind == "object":
                    label(b, b.label, COLORS["object"])
        if self.show_veg.get():
            for b in scene:
                if b.kind == "veg":
                    label(b, b.label, COLORS["veg"])
        if self.show_plates.get():
            for b, text, confirmed, _ in self.plate_view:
                label(b, text if confirmed else f"{text} ?", COLORS["plate"] if confirmed else COLORS["plate_unsure"])
        return image

    def tick(self):
        try:
            self._tick()
        except Exception:  # сбой одного кадра не должен замораживать окно
            traceback.print_exc()
            self.status.configure(text="Сбой при показе кадра. Подробности в окне консоли.")
        if self.args.exit_after and time.monotonic() - self.started > self.args.exit_after:
            self.close()
            return
        self.root.after(15, self.tick)

    def _tick(self):
        self.want_objects, self.want_veg = self.show_objects.get(), self.show_veg.get()
        self.scene_worker.enabled = self.want_objects or self.want_veg
        self.plate_worker.enabled = self.show_plates.get()
        self._update_plates()
        frame = self.camera.frame
        if self.camera.failed and (frame is None or time.monotonic() - self.camera.stamp > 3):
            self.video.configure(image="", text=CAMERA_HELP)
            self.photo = None
        elif frame is not None:
            w, h = max(self.video.winfo_width(), 320), max(self.video.winfo_height(), 240)
            image = self._draw(frame, w, h)
            self.last_canvas = image
            if self.args.exit_after:  # режим проверки: запоминаем, что видели, и последний кадр с овощем и с номером
                veg = [b.label for b in self._active(self.scene_worker) if b.kind == "veg"]
                plate = [text for _, text, confirmed, _ in self.plate_view if confirmed]
                self.seen.update(veg + plate)
                if veg:
                    self.veg_canvas = image
                if plate:
                    self.plate_canvas = image
            self.photo = ImageTk.PhotoImage(image)
            self.video.configure(image=self.photo, text="")
            now = time.monotonic()
            self.ui_rate = 0.9 * self.ui_rate + 0.1 / max(1e-3, now - self.ui_last)
            self.ui_last = now
        self.journal.retry()
        self._status()

    def _status(self):
        scene, plates = self.scene_worker, self.plate_worker
        where = "видеокарта" if scene.on_gpu else "процессор"
        lines = [f"Видео: {self.ui_rate:.0f} кадр/с",
                 f"Предметы/овощи: {scene.rate:.1f} раз/с" if scene.ready else "Предметы/овощи: загрузка…",
                 f"Номера: {plates.rate:.1f} раз/с" if plates.ready else "Номера: загрузка…",
                 f"Считает: {where}"]
        for extra in (scene.error, plates.error, self.journal.warning):
            if extra:
                lines.append(extra)
        self.status.configure(text="\n".join(lines))

    def close(self):
        if self.args.screenshot and self.last_canvas is not None:
            self.last_canvas.save(self.args.screenshot)
            for suffix, canvas in (("овощ", self.veg_canvas), ("номер", self.plate_canvas)):
                if canvas is not None:
                    path = Path(self.args.screenshot)
                    canvas.save(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))
        if self.args.exit_after:
            print("ВИДЕЛ (сколько показов):", dict(self.seen.most_common(25)), flush=True)
            print(f"ИТОГ кадров_с_камеры={self.camera.shots} видео={self.ui_rate:.1f} "
                  f"сцена={self.scene_worker.rate:.1f} номера={self.plate_worker.rate:.1f} "
                  f"gpu={self.scene_worker.on_gpu} ошибки=[{self.scene_worker.error}|{self.plate_worker.error}]", flush=True)
        self.camera.running = False
        self.scene_worker.running = False
        self.plate_worker.running = False
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Распознавание через веб-камеру")
    parser.add_argument("--camera", type=int, default=0, help="номер камеры (по умолчанию 0)")
    parser.add_argument("--source", help="вместо камеры взять фото или видео (для проверки программы)")
    parser.add_argument("--objects-threshold", default=0.5, help="порог уверенности для предметов, 0..1")
    parser.add_argument("--exit-after", type=float, help="закрыться через столько секунд (для проверки)")
    parser.add_argument("--screenshot", help="при закрытии сохранить кадр с рамками сюда (для проверки)")
    args = parser.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
