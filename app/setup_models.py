"""Проверяет, что все модели на месте. Открытые модели докачивает, если их нет (нужен интернет только в этот момент)."""
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
PUBLIC = {
    "dfine-l-obj365.onnx": "https://huggingface.co/onnx-community/dfine_l_obj365-ONNX/resolve/main/onnx/model.onnx",
    "dfine-l-obj365-config.json": "https://huggingface.co/onnx-community/dfine_l_obj365-ONNX/resolve/main/config.json",
    "yolo-v9-s-608-license-plates-end2end.onnx": "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx",
}
# Эти модели обучены специально для программы, скачать их неоткуда: они хранятся только в папке models
TRAINED = ["ru-plate-reader.onnx", "ru-plate-reader.json", "vegetable-classifier.onnx", "vegetable-classifier.json"]


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name, url in PUBLIC.items():
        path = MODELS / name
        if path.exists():
            continue
        print(f"Скачиваю модель {name}…", flush=True)
        temp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temp)
        except OSError as error:
            print(f"Не удалось скачать {name}: {error}\nПроверьте интернет и запустите снова.")
            return 1
        temp.replace(path)
    missing = [n for n in TRAINED if not (MODELS / n).exists()]
    if missing:
        print("Нет файлов обученных моделей:", ", ".join(missing))
        print("Они не скачиваются из интернета. Скопируйте папку models с компьютера, где программа работает.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
