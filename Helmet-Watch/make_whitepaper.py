import argparse, csv, json, statistics as st
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="S Fatima")
ap.add_argument("--dir", default="results")
ap.add_argument("--out", default="Task4_Whitepaper.docx")
a = ap.parse_args()
R = Path(a.dir)
M = json.load(open(R / "metrics.json"))
S = json.load(open(R / "dataset_stats.json"))
PC = list(csv.DictReader(open(R / "per_class.csv")))
FM = (
    list(csv.DictReader(open(R / "frame_metrics.csv")))
    if (R / "frame_metrics.csv").exists()
    else []
)
T, V, SP, C = M["test"], M["validation"], M["speed"], M["config"]

d = Document()
d.styles["Normal"].font.name = "Calibri"
d.styles["Normal"].font.size = Pt(11)
for s in d.sections:
    s.left_margin = s.right_margin = Inches(1)
    s.top_margin = s.bottom_margin = Inches(0.9)


def H(t, lvl=1):
    d.add_heading(t, lvl)


def P(t, italic=False):
    p = d.add_paragraph()
    r = p.add_run(t)
    r.italic = italic
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    return p


def table(head, rows, widths=None):
    t = d.add_table(rows=1, cols=len(head))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(head):
        t.rows[0].cells[i].text = h
        t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    d.add_paragraph()


def fig(name, caption, w=5.6):
    f = R / name
    if f.exists():
        d.add_picture(str(f), width=Inches(w))
        d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = d.add_paragraph(caption)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.runs[0].italic = True
        c.runs[0].font.size = Pt(9.5)


f3 = lambda x: f"{float(x):.3f}"
classes = S["classes"]
tr, va, te = (
    S["splits"].get(k, {"images": 0, "boxes": {}}) for k in ("train", "valid", "test")
)
tot_img = tr["images"] + va["images"] + te["images"]

# ---------- title ----------
t = d.add_heading(
    "Helmet and No Helmet Detection with Adaptive-Threshold Contour Extraction and YOLOv8",
    0,
)
p = d.add_paragraph(f"{a.name}  |  Artificial Intelligence Internship, Progree")
p.runs[0].italic = True

H("Abstract")
P(
    f"This paper presents a real-time computer vision pipeline that detects and counts objects in video streams. "
    f"Two complementary approaches are combined in one web application: (1) a classical OpenCV pipeline that applies "
    f"Gaussian filtering and adaptive thresholding to extract shapes from pixel bounds, and (2) a YOLOv8 detector "
    f"fine-tuned on a {len(classes)}-class traffic dataset ({', '.join(classes)}). Dynamic per-class count vectors are "
    f"overlaid on every frame. On the held-out test split the fine-tuned model reaches mAP@0.5 of {f3(T['mAP50'])}, "
    f"precision {f3(T['precision'])}, recall {f3(T['recall'])} and F1 {f3(T['F1'])}, with a mean inference time of "
    f"{SP['mean_ms']:.1f} ms per image ({SP['fps']:.1f} FPS) on the evaluation hardware."
)

H("1. Introduction")
P(
    "Detecting riders, helmets and no helmets in traffic footage supports road-safety monitoring, for example flagging "
    "riders without helmets. Real-time operation matters: a system that cannot keep up with the video stream cannot "
    "be used for live monitoring. This project therefore evaluates not only accuracy but also frame-by-frame latency."
)
P(
    "The objectives were: build a pipeline that extracts and tracks target contours in real time; apply adaptive "
    "threshold matrices and Gaussian filters to video frames; overlay dynamic text counts per target class; and "
    "evaluate frame-by-frame inference metrics."
)

H("2. System Architecture")
P(
    "The system has three layers. A Web based front end captures frames from an uploaded video or the webcam, "
    "downsizes them to 640 px width and sends each frame as a JPEG to the back end. A Flask back end runs either the "
    "YOLO detector or the contour pipeline and returns bounding boxes, class labels, confidences, per-class counts "
    "and the server-side latency. The front end draws the boxes, renders the count-vector overlay and records "
    "per-frame metrics that can be exported to CSV."
)
H("2.1 Classical contour pipeline", 2)
P(
    "Each frame is converted to grayscale and smoothed with a Gaussian filter to suppress sensor noise. An adaptive "
    "Gaussian threshold then computes a local threshold for every pixel neighbourhood, which makes the segmentation "
    "robust to uneven lighting. External contours are extracted, filtered by a minimum area, and classified by "
    "polygon approximation (triangle, square, rectangle, circle or n-gon). The Gaussian kernel size, adaptive "
    "block size and minimum area are adjustable at run time in the interface."
)
H("2.2 YOLOv8 detector", 2)
P(
    f"The deep-learning branch uses {C['model']} (Ultralytics), a single-stage anchor-free detector. The model was "
    f"initialised from COCO pre-trained weights and fine-tuned for {C['epochs']} epochs at {C['imgsz']} px with batch "
    f"size {C['batch']}. The best checkpoint by validation fitness is loaded by the back end."
)

H("3. Dataset")
P(
    f"The Helmet v9 dataset (Roboflow Universe, CC BY 4.0) contains traffic-camera frames annotated with bounding "
    f"boxes. After removing the ambiguous 'noise' class, {tot_img} images remain across the provided train, "
    f"validation and test splits, which were kept unchanged to avoid leaking near-duplicate frames between splits."
)
rows = [
    [c, tr["boxes"].get(c, 0), va["boxes"].get(c, 0), te["boxes"].get(c, 0)]
    for c in classes
]
rows.append(["Images", tr["images"], va["images"], te["images"]])
table(["Class", "Train boxes", "Valid boxes", "Test boxes"], rows)
P(
    "Table 1. Dataset composition (annotated boxes per class and image counts per split).",
    italic=True,
)
big = max(classes, key=lambda c: tr["boxes"].get(c, 0))
small = min(classes, key=lambda c: tr["boxes"].get(c, 0))
P(
    f"The class distribution is imbalanced: '{big}' has {tr['boxes'].get(big, 0)} training boxes while '{small}' has "
    f"{tr['boxes'].get(small, 0)}. Many frames come from a small number of fixed cameras, so scene diversity is limited."
)

H("4. Experimental Results")
H("4.1 Overall accuracy", 2)
table(
    ["Split", "Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"],
    [
        ["Validation"]
        + [f3(V[k]) for k in ("precision", "recall", "F1", "mAP50", "mAP50-95")],
        ["Test"]
        + [f3(T[k]) for k in ("precision", "recall", "F1", "mAP50", "mAP50-95")],
    ],
)
P(
    "Table 2. Detection metrics of the fine-tuned model (F1 = 2PR/(P+R) computed from mean precision and recall).",
    italic=True,
)
H("4.2 Per-class performance", 2)
table(
    ["Class", "Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"],
    [
        [r["class"], r["precision"], r["recall"], r["F1"], r["mAP50"], r["mAP50-95"]]
        for r in PC
    ],
)
P("Table 3. Per-class metrics on the test split.", italic=True)
best_c = max(PC, key=lambda r: float(r["mAP50"]))
worst_c = min(PC, key=lambda r: float(r["mAP50"]))
low_r = min(PC, key=lambda r: float(r["recall"]))
P(
    f"'{best_c['class']}' is detected best (mAP@0.5 {best_c['mAP50']}), while '{worst_c['class']}' is the weakest "
    f"(mAP@0.5 {worst_c['mAP50']}). The lowest recall belongs to '{low_r['class']}' ({low_r['recall']}), meaning the "
    f"model misses a larger share of those objects than of any other class."
)
fig(
    "results.png", "Figure 1. Training and validation loss and metric curves per epoch."
)
fig(
    "confusion_matrix.png",
    "Figure 2. Confusion matrix (raw counts) on the validation split",
    4.6,
)
fig("PR_curve.png", "Figure 3. Precision-recall curves per class.", 4.6)

H("5. Frame-by-Frame Real-Time Performance")
P(
    f"Offline benchmark: over {SP['images']} test images the model needed {SP['mean_ms']:.1f} ms on average "
    f"({SP['fps']:.1f} FPS) for inference through the Python API."
)
if FM:
    rows = []
    for mode in sorted({r["mode"] for r in FM}):
        sub = [r for r in FM if r["mode"] == mode]
        srv = [float(r["server_ms"]) for r in sub]
        rt = [float(r["roundtrip_ms"]) for r in sub]
        fps = [float(r["fps"]) for r in sub]
        p95 = sorted(rt)[int(0.95 * (len(rt) - 1))]
        rows.append(
            [
                mode,
                len(sub),
                f"{st.mean(srv):.1f}",
                f"{st.mean(rt):.1f}",
                f"{st.median(rt):.1f}",
                f"{p95:.1f}",
                f"{st.mean(fps):.1f}",
                f"{st.mean(int(r['objects']) for r in sub):.1f}",
            ]
        )
    table(
        [
            "Mode",
            "Frames",
            "Server ms",
            "Round-trip ms",
            "Median ms",
            "P95 ms",
            "Mean FPS",
            "Objects/frame",
        ],
        rows,
    )
    P(
        "Table 4. Live metrics logged by the web application (frame_metrics.csv), one row per processed frame.",
        italic=True,
    )
    P(
        "The gap between server time and round-trip time is the cost of JPEG encoding, HTTP transfer and canvas "
        "rendering. For live use, the round-trip figure and the 95th-percentile latency are the relevant numbers, "
        "because they determine whether the overlay keeps up with the video."
    )
else:
    P(
        "No frame_metrics.csv was found, so the live-application table is omitted. Run the web app, process a video "
        "and press 'Frame log' to export it into the results folder.",
        italic=True,
    )

H("6. Discussion and Limitations")
gap = float(V["mAP50"]) - float(T["mAP50"])
P(
    f"Validation mAP@0.5 ({f3(V['mAP50'])}) and test mAP@0.5 ({f3(T['mAP50'])}) differ by {abs(gap):.3f}. "
    f"Because the frames originate from a few fixed cameras, both numbers probably overestimate performance on "
    f"unseen camera angles and lighting; validation on footage from new locations is needed before deployment."
)
P(
    "Counts are per-frame detections, not unique objects: without a tracker, the same rider is counted again in every "
    "frame. Adding a tracker such as ByteTrack would give true cumulative counts. The contour pipeline needs no "
    "training and is fast, but it responds to any edge structure and cannot tell a helmet from other round shapes, "
    "so it is best viewed as a baseline and a teaching tool rather than a detector. Streaming frames to a server adds "
    "network overhead; running the model in the browser or on the capture device would reduce latency. In informal tests on crowded, elevated footage where riders appear small and far from the camera, the model missed several riders, because most training images show riders at closer range and the input is downscaled to 416 px; detection is more reliable on close, clearly visible riders."
)

H("7. Conclusion")
P(
    f"The project delivers an end-to-end pipeline: adaptive-threshold contour extraction, a fine-tuned YOLOv8 "
    f"detector, a live count overlay and exportable frame-level metrics. The model achieves F1 {f3(T['F1'])} and "
    f"mAP@0.5 {f3(T['mAP50'])} on the test split at {SP['fps']:.1f} FPS offline. Future work includes multi-object "
    f"tracking, more diverse training footage, balancing the no_helmet class"
)

H("References")
for r in [
    "Jocher, G., Chaurasia, A., Qiu, J. Ultralytics YOLOv8, 2023. github.com/ultralytics/ultralytics",
    "Bradski, G. The OpenCV Library. Dr. Dobb's Journal of Software Tools, 2000.",
    "Helmet v9 dataset, Roboflow Universe (CC BY 4.0). universe.roboflow.com/helmet-detection-xy5ky/helmet-htftb",
    "Lin, T.-Y. et al. Microsoft COCO: Common Objects in Context. ECCV, 2014.",
]:
    d.add_paragraph(r, style="List Number")

d.save(a.out)
print("Saved", a.out)
