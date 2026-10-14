import argparse, ast, json, re, shutil
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", help="dataset folder (optional: auto-detected if omitted)")
ap.add_argument("--out", default="data_fixed")
ap.add_argument("--drop", nargs="*", default=["noise"])
a = ap.parse_args()

import zipfile


def find_dataset():
    """Look in this folder and its parent for an unzipped YOLO dataset (or a Helmet zip to unzip)."""
    here = Path(__file__).resolve().parent
    roots = [here, here.parent]
    for r in roots:
        for z in r.glob("*.zip"):
            if (
                "helmet" in z.name.lower()
                and "v9" in z.name.lower()
                and not (r / z.stem).exists()
            ):
                print("Unzipping", z.name, "...")
                zipfile.ZipFile(z).extractall(r / z.stem)
    found = []
    for r in roots:
        for y in list(r.glob("*/data.yaml")) + list(r.glob("*/*/data.yaml")):
            d = y.parent
            if (
                (d / "train" / "labels").exists()
                and d.name != "data_fixed"
                and "data_fixed" not in d.parts
            ):
                found.append(d)
    pref = [d for d in found if "v9" in d.name.lower() or "helmet9" in d.name.lower()]
    pick = pref or found
    if len(pick) == 1 or pref:
        return pick[0]
    print(
        "Could not decide which dataset to use. Found:"
        if found
        else "No dataset found next to this script."
    )
    for d in found:
        print("  ", d)
    print('Run:  python prepare_dataset.py --src "full\\path\\to\\dataset_folder"')
    raise SystemExit(1)


src = Path(a.src) if a.src else find_dataset()
if not a.src:
    print("Using dataset:", src)
out = Path(a.out)
txt = (src / "data.yaml").read_text()
names = ast.literal_eval(re.search(r"names:\s*(\[.*?\])", txt, re.S).group(1))
_n = lambda x: {"with_helmet": "helmet", "without_helmet": "no_helmet"}.get(
    x.strip().lower().replace(" ", "_"), x
)
names = [_n(x) for x in names]
keep = [n for n in names if n not in a.drop]
remap = {names.index(n): keep.index(n) for n in keep}
print("classes:", keep)

stats = {"classes": keep, "splits": {}}
for sp in ("train", "valid", "test"):
    li = src / sp / "labels"
    if not li.exists():
        continue
    (out / sp / "images").mkdir(parents=True, exist_ok=True)
    (out / sp / "labels").mkdir(parents=True, exist_ok=True)
    c, n_img = Counter(), 0
    for lb in li.glob("*.txt"):
        img = next(
            (
                p
                for e in (".jpg", ".jpeg", ".png")
                if (p := src / sp / "images" / (lb.stem + e)).exists()
            ),
            None,
        )
        if img is None:
            continue
        rows = []
        for ln in lb.read_text().splitlines():
            v = ln.split()
            if len(v) < 5:
                continue
            try:
                cid, nums = int(v[0]), list(map(float, v[1:]))
            except ValueError:
                continue
            if cid not in remap:
                continue
            if len(nums) > 4:
                xs, ys = nums[0::2], nums[1::2]
                x1, x2, y1, y2 = (
                    max(0, min(xs)),
                    min(1, max(xs)),
                    max(0, min(ys)),
                    min(1, max(ys)),
                )
                nums = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            if nums[2] <= 0 or nums[3] <= 0:
                continue
            rows.append(f"{remap[cid]} " + " ".join(f"{x:.6f}" for x in nums[:4]))
            c[keep[remap[cid]]] += 1
        shutil.copy(img, out / sp / "images" / img.name)
        (out / sp / "labels" / lb.name).write_text("\n".join(rows))
        n_img += 1
    stats["splits"][sp] = {"images": n_img, "boxes": dict(c)}
    print(sp, n_img, "images", dict(c))

(out / "data.yaml").write_text(
    f"path: {out.resolve()}\ntrain: train/images\\nval: valid/images\\ntest: test/images\\nnc: {len(keep)}\nnames: {keep}\n"
)
Path("results").mkdir(exist_ok=True)
json.dump(stats, open("results/dataset_stats.json", "w"), indent=2)
print("saved", out / "data.yaml")
