# Лицензии моделей / Model licenses

Код репозитория — MIT (файл `LICENSE` в корне). Модели — отдельно, по условиям данных, на которых обучены.
Code is MIT (see root `LICENSE`). Models follow the terms of their training data.

## ru-plate-reader.onnx — чтение российских номеров / Russian plate reader

**CC BY 4.0.** Обучена на AUTO.RIA Numberplate Dataset (CC BY 4.0). При использовании указывать:
«Модель обучена на AUTO.RIA Numberplate Dataset, © AUTO.RIA (ria.com), CC BY 4.0».
Trained on the AUTO.RIA Numberplate Dataset (CC BY 4.0). Attribution: "Trained on AUTO.RIA Numberplate Dataset,
© AUTO.RIA (ria.com), CC BY 4.0".

## vegetable-classifier.onnx — определитель овощей / vegetable classifier

Основа — EfficientNet-B0 с весами ImageNet из torchvision (BSD-3-Clause). Данные для обучения:

- cc92yy3344/vegetable (Hugging Face) — Apache-2.0;
- omorros/kitchen-fruit-vegetable-detection (Hugging Face) — CC BY 4.0, © omorros;
- «Fruit and Vegetable Image Recognition» (Kaggle, Kritik Seth; копия Nattakarn на Hugging Face) —
  **лицензия у набора не указана**.

Из-за последнего набора права на эту модель не гарантируются. Предоставляется **«как есть», для
некоммерческого и ознакомительного использования**, с указанием источников выше. Для коммерческого
применения модель нужно переобучить без этого набора (`обучение/train_veg.py` — убрать его из списка данных).

Provided **as is, for non-commercial / evaluation use**, with the attributions above: one training dataset has
no stated license, so rights to this model cannot be guaranteed. For commercial use, retrain without that dataset.

## Скачиваются при первом запуске (в репозиторий не входят) / Downloaded at first launch

| Модель | Источник | Лицензия |
|---|---|---|
| dfine-l-obj365.onnx | onnx-community/dfine_l_obj365-ONNX (D-FINE, обучен на Objects365) | Apache-2.0 |
| yolo-v9-s-608-license-plates-end2end.onnx | ankandrew/open-image-models (YOLOv9) | **GPL-3.0** (YOLOv9) |

GPL-3.0 у YOLOv9 означает: при распространении программы **вместе** с этой моделью вся сборка должна
распространяться на условиях GPL-3.0 (исходники открыты — здесь это выполняется). Код MIT с GPL совместим.
