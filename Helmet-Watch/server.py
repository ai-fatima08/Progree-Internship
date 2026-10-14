import base64, os, time
from collections import Counter

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

try:
    from ultralytics import YOLO

    # Uses best.pt (your helmet model) if present next to this file, else COCO yolov8n.pt.
    W = os.environ.get("YOLO_WEIGHTS") or (
        "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    )
    MODEL = YOLO(W)
    print("Loaded weights:", W)
except Exception as e:  
    MODEL = None
    print("YOLO not loaded:", e)

app = Flask(__name__, static_folder=".")


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ok=True,
        yolo=MODEL is not None,
        classes=list(MODEL.names.values()) if MODEL else [],
    )


def yolo_detect(img, conf):
    res = MODEL(img, conf=conf, imgsz=960, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(
            {
                "label": res.names[int(b.cls)],
                "conf": float(b.conf),
                "box": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return out, None


def shape_name(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    n = len(approx)
    if n == 3:
        return "triangle"
    if n == 4:
        _, _, w, h = cv2.boundingRect(approx)
        return "square" if 0.9 <= w / h <= 1.1 else "rectangle"
    if n > 6:
        return "circle"
    return f"polygon-{n}"


def contour_detect(img, blur, block, min_area, want_mask):
    blur = blur if blur % 2 else blur + 1
    block = max(3, block if block % 2 else block + 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 5
    )
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append({"label": shape_name(c), "conf": None, "box": [x, y, w, h]})
    mask = None
    if want_mask:
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
        mask = base64.b64encode(buf).decode()
    return out, mask


@app.post("/api/detect")
def detect():
    try:
        f = request.files["frame"].read()
        img = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
        mode = request.form.get("mode", "yolo")
        t0 = time.perf_counter()
        if mode == "yolo":
            if MODEL is None:
                return jsonify(error="YOLO not installed; use Shape contours mode"), 500
            dets, mask = yolo_detect(img, float(request.form.get("conf", 0.4)))
        else:
            dets, mask = contour_detect(
                img,
                int(request.form.get("blur", 5)),
                int(request.form.get("block", 21)),
                int(request.form.get("min_area", 400)),
                request.form.get("mask") == "1",
            )
        ms = (time.perf_counter() - t0) * 1000
        return jsonify(
            detections=dets,
            counts=dict(Counter(d["label"] for d in dets)),
            violations=sum(d["label"] == "no_helmet" for d in dets),
            latency_ms=ms,
            mask=mask,
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
