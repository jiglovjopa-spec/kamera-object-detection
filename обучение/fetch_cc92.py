"""Докачать снимки набора Vegetable Image Dataset (Apache-2.0, зеркало cc92yy3344/vegetable) для классов,
которых нет или мало в основном наборе."""
import concurrent.futures as cf, pathlib, time, urllib.parse, urllib.request
ROOT = pathlib.Path(__file__).parent / "data" / "cc92"
BASE = "https://huggingface.co/datasets/cc92yy3344/vegetable/resolve/main/Vegetable%20Images/train/"
PLAN = {"Broccoli": 350, "Pumpkin": 350, "Bean": 300, "Radish": 150, "Tomato": 120, "Potato": 120, "Carrot": 120,
        "Cucumber": 120, "Cabbage": 120, "Cauliflower": 120, "Brinjal": 120, "Capsicum": 120, "Papaya": 60,
        "Bitter_Gourd": 60, "Bottle_Gourd": 60}


def one(a):
    c, i = a
    p = ROOT / c / f"{i:04d}.jpg"
    if p.exists(): return True
    for attempt in range(4):
        try:
            b = urllib.request.urlopen(urllib.request.Request(BASE + f"{urllib.parse.quote(c)}/{i:04d}.jpg", headers={"User-Agent": "kamera-train"}), timeout=40).read()
            p.write_bytes(b); time.sleep(0.25); return True
        except urllib.error.HTTPError as e:
            if e.code == 404: return False
            time.sleep(30 * (attempt + 1))
        except Exception:
            time.sleep(5)
    return False


jobs = []
for c, n in PLAN.items():
    (ROOT / c).mkdir(parents=True, exist_ok=True)
    jobs += [(c, i) for i in range(1, n + 1)]
with cf.ThreadPoolExecutor(3) as ex:
    res = list(ex.map(one, jobs))
print("скачано", sum(res), "из", len(res), flush=True)
