"""Обучение чтения российских номеров на наборе AUTO.RIA (CC BY 4.0). Запускается один раз, в программу не входит."""
import io, json, pathlib, re, sys, time, zipfile
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent; DATA = ROOT / "data"; OUT = ROOT.parent / "models"
ALPHABET = "ABEKMHOPCTYX0123456789_"; SLOTS = 9; H, W = 64, 192
PAT = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
IDX = {c: i for i, c in enumerate(ALPHABET)}


def prepare():
    cache = DATA / "plates.npz"
    if cache.exists():
        d = np.load(cache); return {k: d[k] for k in d.files}
    z = zipfile.ZipFile(DATA / "ru.zip"); names = z.namelist(); out = {}
    for split in ("train", "val", "test"):
        imgs = [n for n in names if f"/{split}/img/" in n and not n.endswith("/")]
        X, Y = np.zeros((len(imgs), H, W), np.uint8), np.zeros((len(imgs), SLOTS), np.int64); k = bad = 0
        for n in imgs:
            ann = n.replace("/img/", "/ann/").rsplit(".", 1)[0] + ".json"
            text = json.loads(z.read(ann)).get("description", "")
            if not PAT.match(text): bad += 1; continue
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None: bad += 1; continue
            X[k] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            Y[k] = [IDX[c] for c in text.ljust(SLOTS, "_")]; k += 1
        print(split, "взято", k, "отброшено (другой формат)", bad, flush=True)
        out[f"X{split}"], out[f"Y{split}"] = X[:k], Y[:k]
    np.savez(cache, **out); return out


class Res(nn.Module):
    def __init__(s, a, b, st=1):
        super().__init__()
        s.c1 = nn.Conv2d(a, b, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(b)
        s.c2 = nn.Conv2d(b, b, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(b)
        s.sk = nn.Identity() if a == b and st == 1 else nn.Sequential(nn.Conv2d(a, b, 1, st, bias=False), nn.BatchNorm2d(b))
    def forward(s, x): return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class PlateNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(1, 24, 3, 1, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
                            Res(24, 32, 2), Res(32, 32), Res(32, 64, 2), Res(64, 64), Res(64, 128, 2), Res(128, 128),
                            Res(128, 192, 2), Res(192, 192), nn.Conv2d(192, 48, 1))
        s.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(48 * 4 * 12, 512), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(512, SLOTS * len(ALPHABET)))
    def forward(s, x): return s.h(s.f(x)).view(-1, SLOTS, len(ALPHABET))


class Export(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return F.softmax(s.m(x), dim=-1)


def augment(x):
    """Портит чистый номер так, как его испортит веб-камера: поворот, ракурс, сдвиг рамки, размытие, шум, свет."""
    n = x.size(0); d = x.device; r = lambda a, b: torch.empty(n, device=d).uniform_(a, b)
    ang, sx, sy, shx, shy = r(-.14, .14), r(.85, 1.32), r(.85, 1.45), r(-.28, .28), r(-.1, .1)
    tx, ty = r(-.1, .1), r(-.14, .14)
    th = torch.stack([torch.stack([sx * torch.cos(ang), shx - torch.sin(ang), tx], 1),
                      torch.stack([torch.sin(ang) + shy, sy * torch.cos(ang), ty], 1)], 1)
    g = F.affine_grid(th, x.size(), align_corners=False)
    g = g + (g[..., :1] * g[..., 1:]) * r(-.12, .12).view(-1, 1, 1, 1)  # лёгкая перспектива
    # Заполнение вне кадра — краем: как фон вокруг номера
    x = F.grid_sample(x, g, padding_mode="border", align_corners=False)
    s = float(torch.empty(1).uniform_(.3, 1.0))
    if s < .9:  # камера видит номер мелко
        x = F.interpolate(F.interpolate(x, scale_factor=s, mode="area"), size=(H, W), mode="bilinear", align_corners=False)
    x = (x * r(.55, 1.25).view(-1, 1, 1, 1) + r(-.15, .15).view(-1, 1, 1, 1)).clamp(0, 1)
    x = x.clamp(1e-3, 1) ** torch.exp(r(-.6, .6)).view(-1, 1, 1, 1)
    x = x + torch.randn_like(x) * r(0, .06).view(-1, 1, 1, 1)
    # Блик или тень полосой
    if torch.rand(1) < .3:
        c = torch.rand(n, device=d).view(-1, 1, 1, 1) * W; ww = torch.empty(n, device=d).uniform_(8, 50).view(-1, 1, 1, 1)
        xs = torch.arange(W, device=d).float().view(1, 1, 1, -1)
        x = x + torch.exp(-((xs - c) / ww) ** 2) * r(-.45, .45).view(-1, 1, 1, 1)
    return x.clamp(0, 1)


@torch.no_grad()
def accuracy(m, X, Y, dev):
    m.eval(); ok = tot = 0
    for i in range(0, len(X), 512):
        x = torch.from_numpy(X[i:i + 512]).to(dev).float()[:, None] / 255
        p = m(x).argmax(-1).cpu().numpy(); ok += (p == Y[i:i + 512]).all(1).sum(); tot += len(p)
    m.train(); return ok / tot


def main():
    EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    d = prepare(); dev = "cuda"
    Xt, Yt = torch.from_numpy(d["Xtrain"]), torch.from_numpy(d["Ytrain"])
    print("обучение на", len(Xt), flush=True)
    m = PlateNet().to(dev); print("параметров", sum(p.numel() for p in m.parameters()), flush=True)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4); BS = 128  # 256 не помещается в 4 ГБ видеопамяти
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * (len(Xt) // BS))
    best = 0
    for ep in range(EPOCHS):
        t = time.time(); perm = torch.randperm(len(Xt)); tl = 0
        for i in range(len(Xt) // BS):
            ix = perm[i * BS:(i + 1) * BS]
            x = augment(Xt[ix].to(dev).float()[:, None] / 255); y = Yt[ix].to(dev)
            loss = F.cross_entropy(m(x).reshape(-1, len(ALPHABET)), y.reshape(-1), label_smoothing=.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tl += loss.item()
        va = accuracy(m, d["Xval"], d["Yval"], dev)
        print(f"эпоха {ep + 1}/{EPOCHS} потери {tl / (len(Xt) // BS):.3f} val {va:.2%} {time.time() - t:.0f}с", flush=True)
        if va >= best:
            best = va; torch.save(m.state_dict(), ROOT / "plate_best.pt")
    m.load_state_dict(torch.load(ROOT / "plate_best.pt")); print(f"ТЕСТ (номера, не участвовавшие в обучении): {accuracy(m, d['Xtest'], d['Ytest'], dev):.2%}", flush=True)
    e = Export(m).eval().cpu()
    torch.onnx.export(e, torch.zeros(1, 1, H, W), OUT / "ru-plate-reader.onnx", input_names=["input"], output_names=["plate"],
                      dynamic_axes={"input": {0: "n"}, "plate": {0: "n"}}, opset_version=17, dynamo=False)
    (OUT / "ru-plate-reader.json").write_text(json.dumps({"alphabet": ALPHABET, "slots": SLOTS, "height": H, "width": W}), encoding="utf-8")
    print("сохранено", flush=True)


if __name__ == "__main__":
    main()
