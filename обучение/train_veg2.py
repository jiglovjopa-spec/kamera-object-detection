"""Дообучение определителя овоща: порча как у веб-камеры и настоящие снимки с кухни (чётные снимки; нечётные остаются для проверки)."""
import collections, io, json, pathlib, random, shutil, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv
from PIL import Image
from train_veg import DS, Export, MAP, OUT, ROOT, load

KITCHEN = pathlib.Path(sys.argv[2])
K_MAP = {"bell_pepper_green": "болгарский перец", "bell_pepper_red": "болгарский перец", "carrot": "морковь", "cucumber": "огурец",
         "onion": "лук", "potato": "картофель", "tomato": "помидор"}


def kitchen_crops():
    ann = json.loads((KITCHEN / "annotations.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}; files = {i["id"]: i["file_name"] for i in ann["images"]}; out = []
    for a in ann["annotations"]:
        if a["image_id"] % 2:
            continue  # нечётные снимки не трогаем
        im = cv2.imdecode(np.fromfile(KITCHEN / pathlib.Path(files[a["image_id"]]).name, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        x, y, w, h = a["bbox"]; px, py = .08 * w, .08 * h
        c = im[max(0, int(y - py)):int(y + h + py), max(0, int(x - px)):int(x + w + px)]
        if c.size == 0:
            continue
        out.append((cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), K_MAP.get(cats[a["category_id"]], "другое"), "kitchen"))
    return out


class Webcam:
    """Что делает камера ноутбука: другой баланс белого, бледные цвета, смаз движения, шум, сжатие."""
    def __call__(s, im):
        a = np.asarray(im).astype(np.float32)
        if random.random() < .7:  # баланс белого и выцветание
            a = a * np.array([random.uniform(.8, 1.2) for _ in range(3)], np.float32)
            g = a.mean(2, keepdims=True); a = g + (a - g) * random.uniform(.45, 1.05)
            a = 255 * np.clip(a / 255, 1e-3, 1) ** random.uniform(.6, 1.4)
        a = np.clip(a, 0, 255).astype(np.uint8)
        if random.random() < .35:  # смаз движения
            L = random.choice([5, 7, 9, 13, 17]); k = np.zeros((L, L), np.float32); k[L // 2] = 1
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2 - .5, L / 2 - .5), random.uniform(0, 180), 1), (L, L)); k /= max(k.sum(), 1e-3)
            a = cv2.filter2D(a, -1, k)
        if random.random() < .5:  # мелкий предмет и сжатие
            h, w = a.shape[:2]; f = random.uniform(.35, .9); a = cv2.resize(cv2.resize(a, (max(8, int(w * f)), max(8, int(h * f)))), (w, h))
            a = cv2.imdecode(cv2.imencode(".jpg", a[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 85)])[1], cv2.IMREAD_COLOR)[..., ::-1]
        if random.random() < .5:
            a = np.clip(a + np.random.normal(0, random.uniform(2, 9), a.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(a.astype(np.uint8))


def main():
    EPOCHS = int(sys.argv[1]); items = load(); kit = kitchen_crops()
    print("снимков с кухни", len(kit), collections.Counter(i[1] for i in kit), flush=True)
    random.seed(3); random.shuffle(items); random.shuffle(kit); by = collections.defaultdict(list)
    for it in items + kit: by[it[1]].append(it)
    classes = sorted(by); tr, va = [], []
    for c, lst in by.items():
        random.shuffle(lst); n = max(8, len(lst) // 7); va += lst[:n]; tr += lst[n:]
    w = collections.Counter(i[1] for i in tr)
    # Снимки с кухни ближе всего к настоящей камере: показываем их в 4 раза чаще
    sampler = torch.utils.data.WeightedRandomSampler([(4 if i[2] == "kitchen" else 1) / w[i[1]] for i in tr], 5000)
    ds = DS(tr, classes, True); M, S = [.485, .456, .406], [.229, .224, .225]
    ds.tf = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224, scale=(.5, 1.)), tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomRotation(25),
                                   tv.transforms.ColorJitter(.3, .3, .3, .03), Webcam(), tv.transforms.ToTensor(), tv.transforms.Normalize(M, S)])
    L = torch.utils.data.DataLoader(ds, 32, sampler=sampler, drop_last=True); V = torch.utils.data.DataLoader(DS(va, classes, False), 64)
    old = json.loads((OUT / "vegetable-classifier.json").read_text(encoding="utf-8"))["classes"]; assert old == classes, (old, classes)
    shutil.copy(OUT / "vegetable-classifier.onnx", ROOT / "vegetable-classifier-v1.onnx")
    dev = "cuda"; m = tv.models.efficientnet_b0(); m.classifier[1] = nn.Linear(1280, len(classes)); m.load_state_dict(torch.load(ROOT / "veg_best.pt")); m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EPOCHS * len(L), pct_start=.15); best = 0
    for ep in range(EPOCHS):
        t = time.time(); m.train()
        for x, y in L:
            loss = F.cross_entropy(m(x.to(dev)), y.to(dev), label_smoothing=.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        m.eval(); ok = tot = 0
        with torch.no_grad():
            for x, y in V:
                ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); tot += len(y)
        print(f"эпоха {ep + 1}/{EPOCHS} проверка {ok / tot:.2%} {time.time() - t:.0f}с", flush=True)
        if ok / tot >= best:
            best = ok / tot; torch.save(m.state_dict(), ROOT / "veg2_best.pt")
    m.load_state_dict(torch.load(ROOT / "veg2_best.pt"))
    torch.onnx.export(Export(m).eval().cpu(), torch.zeros(1, 3, 224, 224), OUT / "vegetable-classifier.onnx", input_names=["input"],
                      output_names=["probs"], dynamic_axes={"input": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    print("сохранено", flush=True)


if __name__ == "__main__":
    main()
