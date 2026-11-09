"""Распознаватели: предметы и овощи (одна модель + уточнение овоща), автомобильные номера. Всё считается локально."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)  # без технических предупреждений в окне консоли
if hasattr(ort, "preload_dlls"):  # загрузить библиотеки NVIDIA из окружения программы
    try:
        ort.preload_dlls()
    except Exception as error:  # без них программа просто перейдёт на процессор
        print("Библиотеки NVIDIA не загрузились, считаем на процессоре:", error)

from open_image_models.detection.core.yolo_v9.inference import YoloV9Detector  # noqa: E402

from labels import NOT_VEGETABLES, OBJECTS_RU, VEGETABLE_CANDIDATES  # noqa: E402
from russian_plate import PlateReading, decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float
    kind: str  # object | plate | veg
    reading: PlateReading | None = None


def _options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return so


def _session(path: Path, gpu: bool = True) -> ort.InferenceSession:
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    providers = [p for p in wanted if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), _options(), providers=providers)


def uses_gpu(session: ort.InferenceSession) -> bool:
    return session.get_providers()[0] == "CUDAExecutionProvider"


class VegetableClassifier:
    """Наша модель: смотрит на вырезанную рамку и называет овощ."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self.session = _session(MODELS / "vegetable-classifier.onnx")
        self.classes = json.loads((MODELS / "vegetable-classifier.json").read_text(encoding="utf-8"))["classes"]

    def __call__(self, crops_rgb: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops_rgb:
            return []
        batch = np.stack([((cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - self.MEAN)
                           / self.STD).transpose(2, 0, 1) for c in crops_rgb])
        probs = self.session.run(None, {"input": batch})[0]
        return [(self.classes[int(p.argmax())], float(p.max())) for p in probs]


class SceneDetector:
    """D-FINE (365 видов предметов, в том числе овощи) + уточнение овоща."""

    def __init__(self, object_threshold: float = 0.5, candidate_threshold: float = 0.3, veg_threshold: float = 0.55,
                 fruit_override_threshold: float = 0.85):
        self.session = _session(MODELS / "dfine-l-obj365.onnx")
        self.on_gpu = uses_gpu(self.session)
        config = json.loads((MODELS / "dfine-l-obj365-config.json").read_text(encoding="utf-8"))
        self.names = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self.candidate = np.array([n in VEGETABLE_CANDIDATES for n in self.names])
        self.object_threshold, self.candidate_threshold, self.veg_threshold = object_threshold, candidate_threshold, veg_threshold
        self.fruit_override_threshold = fruit_override_threshold
        self.not_vegetable = np.array([n in NOT_VEGETABLES for n in self.names])
        self.classifier = VegetableClassifier()

    def raw(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = cv2.cvtColor(cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        logits, boxes = self.session.run(["logits", "pred_boxes"], {"pixel_values": x})
        return 1.0 / (1.0 + np.exp(-logits[0])), boxes[0]

    def __call__(self, frame_bgr: np.ndarray, want_objects: bool = True, want_vegetables: bool = True) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        probs, boxes = self.raw(frame_bgr)
        probs[:, 0] = 0  # пустой класс «None»
        best = probs.argmax(axis=1)
        top = probs.max(axis=1)
        # Для овощных классов порог ниже: дальше их всё равно проверит определитель овоща
        cand_score = (probs * self.candidate).max(axis=1)
        cand_class = (probs * self.candidate).argmax(axis=1)
        keep = np.where((top >= self.object_threshold) | (cand_score >= self.candidate_threshold))[0]
        if keep.size == 0:
            return []
        cx, cy, bw, bh = boxes[keep].T
        rects = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h], axis=1)
        scores = np.maximum(top[keep], cand_score[keep])
        chosen = np.array(cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), 0.01, 0.6)).reshape(-1)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results: list[Box] = []
        to_classify: list[tuple[np.ndarray, tuple]] = []
        for i in chosen:
            q = keep[i]
            x, y, bw_, bh_ = rects[i]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw_)), min(h, int(y + bh_))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            is_candidate = cand_score[q] >= self.candidate_threshold
            if is_candidate and want_vegetables:
                px, py = int(0.08 * (x2 - x1)), int(0.08 * (y2 - y1))
                crop = rgb[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
                to_classify.append((crop, (x1, y1, x2, y2, q)))
            elif top[q] >= self.object_threshold and want_objects:
                name = self.names[int(best[q])]
                results.append(Box(x1, y1, x2, y2, OBJECTS_RU.get(name, name), float(top[q]), "object"))

        verdicts = self.classifier([c for c, _ in to_classify])
        for (_, (x1, y1, x2, y2, q)), (veg, confidence) in zip(to_classify, verdicts):
            # Основная модель видит яблоко или другой фрукт: на смазанном кадре определитель легко ошибается,
            # поэтому ему нужна большая уверенность (а свёклу или репу основная модель сама не знает — их оставляем)
            needed = self.fruit_override_threshold if self.not_vegetable[cand_class[q]] else self.veg_threshold
            if veg != "другое" and confidence >= needed:
                results.append(Box(x1, y1, x2, y2, veg, confidence, "veg"))
            elif want_objects and top[q] >= self.object_threshold:
                # Определитель не уверен или это не овощ (например, яблоко): оставляем название основной модели
                name = self.names[int(best[q])]
                results.append(Box(x1, y1, x2, y2, OBJECTS_RU.get(name, name), float(top[q]), "object"))
        return results


class PlateReader:
    """Наша модель чтения: обучена на российских номерах. Принимает вырезанный номер в серых тонах."""

    def __init__(self):
        cfg = json.loads((MODELS / "ru-plate-reader.json").read_text(encoding="utf-8"))
        self.alphabet, self.h, self.w = cfg["alphabet"], cfg["height"], cfg["width"]
        # Маленькая модель, на процессоре считается за миллисекунды
        self.session = _session(MODELS / "ru-plate-reader.onnx", gpu=False)

    def read(self, crop_gray: np.ndarray) -> PlateReading:
        x = cv2.resize(crop_gray, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255
        return decode(self.session.run(None, {"input": x})[0][0], self.alphabet)


class PlateDetector:
    def __init__(self, threshold: float = 0.35):
        self.net = YoloV9Detector(MODELS / "yolo-v9-s-608-license-plates-end2end.onnx", ("License Plate",),
                                  conf_thresh=threshold,
                                  providers=[p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                             if p in ort.get_available_providers()],
                                  sess_options=_options())
        self.reader = PlateReader()

    def __call__(self, frame_bgr: np.ndarray) -> list[Box]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        results = []
        for d in self.net.predict(frame_bgr):
            b = d.bounding_box
            px, py = int(0.02 * (b.x2 - b.x1)), int(0.05 * (b.y2 - b.y1))
            x1, y1, x2, y2 = max(0, b.x1 - px), max(0, b.y1 - py), min(w, b.x2 + px), min(h, b.y2 + py)
            if x2 - x1 < 16 or y2 - y1 < 6:
                continue
            reading = self.reader.read(gray[y1:y2, x1:x2])
            results.append(Box(b.x1, b.y1, b.x2, b.y2, reading.text, float(d.confidence), "plate", reading))
        return results
