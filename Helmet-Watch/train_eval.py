import argparse, csv, json, shutil, time
from pathlib import Path

from ultralytics import YOLO


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def summarize(m):
    p, r = float(m.box.mp), float(m.box.mr)
    return {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": p,
        "recall": r,
        "F1": f1(p, r),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_fixed/data.yaml")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--base", default="yolov8n.pt")
ap.add_argument(
    "--fraction",
    type=float,
    default=0.5,
    help="use only this fraction of train images (e.g. 0.5) to train faster",
)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--device", default="", help="'' = auto (GPU if available), or 'cpu'")
ap.add_argument(
    "--resume", action="store_true", help="continue an interrupted training run"
)

ap.add_argument("--skip-train", action="store_true")
ap.add_argument("--run", default="runs/detect/runs/helmet")

a = ap.parse_args()
Path("results").mkdir(exist_ok=True)
if a.skip_train:
    pass
elif a.resume:
    YOLO("runs/detect/runs/helmet/weights/last.pt").train(resume=True)
else:
    YOLO(a.base).train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        fraction=a.fraction,
        workers=a.workers,
        device=a.device or None,
        patience=15,
        project="runs/detect",
        name="helmet",
        exist_ok=True,
    )

run = Path(a.run)
best = YOLO(str(run / "weights" / "best.pt"))
shutil.copy(run / "weights" / "best.pt", "best.pt")

val = best.val(data=a.data, split="val", imgsz=a.imgsz, verbose=False)
test = best.val(data=a.data, split="test", imgsz=a.imgsz, verbose=False)

with open("results/per_class.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "precision", "recall", "F1", "mAP50", "mAP50-95"])
    for i, ci in enumerate(test.box.ap_class_index):
        p, r = float(test.box.p[i]), float(test.box.r[i])
        w.writerow(
            [
                test.names[int(ci)],
                f"{p:.3f}",
                f"{r:.3f}",
                f"{f1(p, r):.3f}",
                f"{float(test.box.ap50[i]):.3f}",
                f"{float(test.box.ap[i]):.3f}",
            ]
        )

imgs = sorted(Path(a.data).parent.glob("test/images/*"))[:100]
best(str(imgs[0]), verbose=False)  # warm-up
ms = []
for p in imgs:
    t = time.perf_counter()
    best(str(p), verbose=False)
    ms.append((time.perf_counter() - t) * 1000)
mean = sum(ms) / len(ms)

json.dump(
    {
        "validation": summarize(val),
        "test": summarize(test),
        "speed": {"images": len(ms), "mean_ms": mean, "fps": 1000 / mean},
        "config": {
            "model": a.base,
            "epochs": a.epochs,
            "imgsz": a.imgsz,
            "batch": a.batch,
        },
    },
    open("results/metrics.json", "w"),
    indent=2,
)
for n in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"):
    if (run / n).exists():
        shutil.copy(run / n, Path("results") / n)
print(open("results/metrics.json").read())
print(
    "\nDone. best.pt is ready. Next: python server.py (test with videos), then python make_whitepaper.py"
)
