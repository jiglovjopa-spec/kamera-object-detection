"""Обучение определителя овоща по вырезанной рамке. Запускается один раз, в программу не входит."""
import collections, glob, hashlib, io, json, pathlib, random, sys, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv
from PIL import Image

ROOT = pathlib.Path(__file__).parent; DATA = ROOT / "data"; OUT = ROOT.parent / "models"
MAP = {  # название в наборах -> русское имя. "другое" — не овощ, чтобы яблоко не называлось помидором
 "tomato": "помидор", "cucumber": "огурец", "potato": "картофель", "carrot": "морковь", "onion": "лук", "cabbage": "капуста",
 "bell pepper": "болгарский перец", "capsicum": "болгарский перец", "paprika": "болгарский перец", "chilli pepper": "острый перец",
 "jalepeno": "острый перец", "eggplant": "баклажан", "brinjal": "баклажан", "garlic": "чеснок", "beetroot": "свёкла",
 "cauliflower": "цветная капуста", "broccoli": "брокколи", "pumpkin": "тыква", "raddish": "редис", "radish": "редис",
 "corn": "кукуруза", "sweetcorn": "кукуруза", "lettuce": "салат", "ginger": "имбирь", "peas": "горох", "sweetpotato": "батат",
 "spinach": "шпинат", "turnip": "репа", "bean": "стручковая фасоль", "apple": "другое", "banana": "другое", "grapes": "другое",
 "kiwi": "другое", "lemon": "другое", "mango": "другое", "orange": "другое", "pear": "другое", "pineapple": "другое",
 "pomegranate": "другое", "watermelon": "другое", "soy beans": "другое", "papaya": "другое", "bitter_gourd": "другое", "bottle_gourd": "другое"}


def load():
    items, seen = [], set()
    for f in sorted(glob.glob(str(DATA / "*.parquet"))):
        t = pq.read_table(f); names = t.schema.field("label").type if False else None
        meta = json.loads(t.schema.metadata[b"huggingface"].decode())["info"]["features"]["label"]["names"]
        for rec in t.to_pylist():
            b = rec["image"]["bytes"]; h = hashlib.md5(b).hexdigest()
            if h in seen: continue
            seen.add(h)
            lab = meta[rec["label"]].lower().replace(" ", "_") if meta[rec["label"]].lower() not in MAP else meta[rec["label"]].lower()
            ru = MAP.get(lab) or MAP.get(lab.replace("_", " "))
            if ru: items.append((b, ru, f))
    for f in sorted(glob.glob(str(DATA / "cc92" / "*" / "*.jpg"))):
        ru = MAP.get(pathlib.Path(f).parent.name.lower())
        if ru: items.append((pathlib.Path(f).read_bytes(), ru, f))
    print("снимков без повторов", len(items), collections.Counter(i[1] for i in items), flush=True)
    return items


class DS(torch.utils.data.Dataset):
    def __init__(s, items, classes, train):
        s.items, s.cl, s.train = items, {c: i for i, c in enumerate(classes)}, train
        M, S = [.485, .456, .406], [.229, .224, .225]
        s.tf = tv.transforms.Compose(([
            tv.transforms.RandomResizedCrop(224, scale=(.45, 1.)), tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(.2),
            tv.transforms.RandomRotation(25), tv.transforms.ColorJitter(.35, .35, .3, .03), tv.transforms.RandomApply([tv.transforms.GaussianBlur(5)], .2)]
            if train else [tv.transforms.Resize((224, 224))]) + [tv.transforms.ToTensor(), tv.transforms.Normalize(M, S)])
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        b, ru, _ = s.items[i]
        return s.tf(Image.open(io.BytesIO(b)).convert("RGB")), s.cl[ru]


class Export(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return F.softmax(s.m(x), dim=-1)


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    items = load(); random.seed(3); random.shuffle(items)
    by = collections.defaultdict(list)
    for it in items: by[it[1]].append(it)
    classes = sorted(by); tr, va = [], []
    for c, lst in by.items():
        n = max(8, len(lst) // 7); va += lst[:n]; tr += lst[n:]
    print("классов", len(classes), "обучение", len(tr), "проверка", len(va), flush=True)
    w = collections.Counter(i[1] for i in tr)
    sampler = torch.utils.data.WeightedRandomSampler([1 / w[i[1]] for i in tr], 5000)
    L = torch.utils.data.DataLoader(DS(tr, classes, True), 32, sampler=sampler, num_workers=0, drop_last=True)
    V = torch.utils.data.DataLoader(DS(va, classes, False), 64, num_workers=0)
    dev = "cuda"
    m = tv.models.efficientnet_b0(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(1280, len(classes)); m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), 6e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-4, total_steps=EPOCHS * len(L)); best = 0
    for ep in range(EPOCHS):
        t = time.time(); m.train()
        for x, y in L:
            loss = F.cross_entropy(m(x.to(dev)), y.to(dev), label_smoothing=.1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        m.eval(); ok = tot = 0; per = collections.Counter(); pert = collections.Counter()
        with torch.no_grad():
            for x, y in V:
                p = m(x.to(dev)).argmax(1).cpu(); ok += (p == y).sum().item(); tot += len(y)
                for a, b in zip(p.tolist(), y.tolist()): pert[b] += 1; per[b] += a == b
        print(f"эпоха {ep + 1}/{EPOCHS} проверка {ok / tot:.2%} {time.time() - t:.0f}с", flush=True)
        if ok / tot >= best:
            best = ok / tot; torch.save(m.state_dict(), ROOT / "veg_best.pt"); bestper = {classes[k]: f"{per[k]}/{pert[k]}" for k in pert}
    print("лучшая", best, bestper, flush=True)
    m.load_state_dict(torch.load(ROOT / "veg_best.pt"))
    torch.onnx.export(Export(m).eval().cpu(), torch.zeros(1, 3, 224, 224), OUT / "vegetable-classifier.onnx", input_names=["input"],
                      output_names=["probs"], dynamic_axes={"input": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    (OUT / "vegetable-classifier.json").write_text(json.dumps({"classes": classes}, ensure_ascii=False), encoding="utf-8")
    print("сохранено", flush=True)


if __name__ == "__main__":
    main()
