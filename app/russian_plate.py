"""Разбор ответа модели чтения номера под российский формат.

Обычный российский номер: буква, 3 цифры, 2 буквы, код региона из 2 или 3 цифр (А123ВС77, А123ВС777).
Буквы только те 12, что совпадают по начертанию с латинскими: А В Е К М Н О Р С Т У Х.

Для каждой позиции берём самый вероятный символ из разрешённых там. Это исправляет ошибки
вида «0 вместо О» и «8 вместо В», потому что цифра и буква стоят на разных местах.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LATIN_TO_CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
CYR_TO_LATIN = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"
PAD = "_"  # заполнитель пустых позиций в конце номера
TEMPLATES = ("LDDDLLDD", "LDDDLLDDD")


@dataclass(frozen=True)
class PlateReading:
    text: str  # кириллицей, например А123ВС77
    confidence: float  # средняя уверенность по символам, 0..1
    weakest: float  # уверенность в самом сомнительном символе
    raw: str  # что прочитала модель без поправки на формат


def decode(probs: np.ndarray, alphabet: str) -> PlateReading:
    """probs — вероятности размером (позиции, символы) для одного номера."""
    probs = np.asarray(probs, dtype=np.float64)
    raw = "".join(alphabet[i] for i in probs.argmax(axis=1)).rstrip(PAD)
    index = {ch: i for i, ch in enumerate(alphabet)}
    letters = [index[c] for c in LETTERS]
    digits = [index[c] for c in DIGITS]
    pad = index[PAD]

    best: tuple[float, str, list[float]] | None = None
    for template in TEMPLATES:
        logp, chars, conf = 0.0, [], []
        for pos, kind in enumerate(template):
            allowed = letters if kind == "L" else digits
            j = allowed[int(probs[pos, allowed].argmax())]
            logp += float(np.log(probs[pos, j] + 1e-6))
            chars.append(alphabet[j])
            conf.append(float(probs[pos, j]))
        for pos in range(len(template), probs.shape[0]):
            logp += float(np.log(probs[pos, pad] + 1e-6))
        if best is None or logp > best[0]:
            best = (logp, "".join(chars), conf)

    _, text, conf = best
    return PlateReading(text.translate(LATIN_TO_CYR), float(np.mean(conf)), float(np.min(conf)), raw)


def to_latin(text: str) -> str:
    return text.translate(CYR_TO_LATIN)
