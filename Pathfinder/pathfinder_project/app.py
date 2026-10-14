import csv
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from pathfinder.algorithms import ALGORITHMS
from pathfinder.dataset import (
    CATEGORIES,
    default_data_dir,
    default_endpoints,
    index_maps,
    load_any,
    load_moving_ai,
    make_scenarios,
    window_scenario,
)

ROOT = Path(__file__).parent
app = Flask(__name__, static_folder=None)
DATA_DIR = default_data_dir()
INDEX = index_maps(DATA_DIR)
BY_NAME = {r["name"]: r for r in INDEX}
Q_MAX_FREE = 2500  # tabular Q-Learning is capped so a request cannot run for minutes


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.errorhandler(ValueError)
def bad(e):
    return jsonify({"error": str(e)}), 400


# helpers
@lru_cache(maxsize=12)
def cached_grid(name):
    return load_moving_ai(BY_NAME[name]["path"])


@lru_cache(maxsize=24)
def cached_scenarios(name):
    return make_scenarios(cached_grid(name), 5, seed=7)


def rows_of(grid):
    return ["".join(map(str, r)) for r in grid.tolist()]


def to_grid(payload):
    if "rows" in payload:
        g = np.array(
            [[1 if ch == "1" else 0 for ch in row] for row in payload["rows"]],
            dtype=np.uint8,
        )
    else:
        g = np.array(payload["grid"], dtype=np.uint8)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2-D array")
    s, t = payload.get("start"), payload.get("goal")
    if s is None or t is None:
        s, t = default_endpoints(g)
    s, t = tuple(s), tuple(t)
    for name, p in (("start", s), ("goal", t)):
        if not (0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]) or g[p]:
            raise ValueError(f"{name} {list(p)} is outside the grid or on a wall")
    return g, s, t


def pack(res, C):
    out = {
        k: v
        for k, v in res.items()
        if k not in ("curve", "episode_steps", "order", "path")
    }
    out["path"] = [r * C + c for r, c in res["path"]]
    out["order"] = [r * C + c for r, c in res["order"]]
    if "episode_steps" in res:
        steps = [st for st, ok in res["episode_steps"] if ok]
        out["learning_curve"] = steps[:: max(1, len(steps) // 200)]
    return out


def run(algo, g, s, t):
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algo}")
    free = int((g == 0).sum())
    if algo == "Q-Learning" and free > Q_MAX_FREE:
        return {
            "algorithm": algo,
            "found": False,
            "skipped": True,
            "path": [],
            "order": [],
            "path_len": 0,
            "expanded_nodes": 0,
            "expanded_density": 0,
            "runtime_ms": 0,
            "free_cells": free,
            "message": f"Tabular Q-Learning is limited to {Q_MAX_FREE} free cells (this grid has {free}). "
            f"Load a window crop of the map instead.",
        }
    kw = {"time_limit": 25.0} if algo == "Q-Learning" else {}
    return pack(ALGORITHMS[algo](g, s, t, **kw), g.shape[1])


# routes
@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/dataset")
def dataset_info():
    if not INDEX:
        return jsonify({"available": False, "folder": str(DATA_DIR)})
    counts = Counter(r["category"] for r in INDEX)
    return jsonify(
        {
            "available": True,
            "folder": str(DATA_DIR),
            "count": len(INDEX),
            "categories": [
                {"id": c, "label": CATEGORIES[c], "count": counts.get(c, 0)}
                for c in CATEGORIES
                if counts.get(c)
            ],
        }
    )


@app.get("/api/maps")
def maps():
    cat = request.args.get("category")
    return jsonify(
        [
            {
                "name": r["name"],
                "category": r["category"],
                "height": r["height"],
                "width": r["width"],
            }
            for r in INDEX
            if not cat or r["category"] == cat
        ]
    )


@app.get("/api/map")
def get_map():
    name = request.args.get("name", "")
    if name not in BY_NAME:
        raise ValueError(f"unknown map '{name}'")
    mode = request.args.get("mode", "window")
    k = int(request.args.get("scenario", 0)) % 5
    grid = cached_grid(name)
    if mode == "full":
        s, t, d = cached_scenarios(name)[k]
        g, origin = grid, (0, 0)
    else:
        size = max(10, min(int(request.args.get("size", 40)), 48))
        g, origin, s, t, d = window_scenario(grid, size, seed=11 + k)
    return jsonify(
        {
            "name": name,
            "category": BY_NAME[name]["category"],
            "mode": mode,
            "rows": g.shape[0],
            "cols": g.shape[1],
            "grid": rows_of(g),
            "start": list(s),
            "goal": list(t),
            "bfs_dist": d,
            "origin": list(origin),
            "map_height": grid.shape[0],
            "map_width": grid.shape[1],
            "wall_pct": round(float(g.mean()) * 100, 1),
        }
    )


@app.post("/api/solve")
def solve():
    d = request.get_json(force=True)
    g, s, t = to_grid(d)
    return jsonify(run(d.get("algo", "A*"), g, s, t))


@app.post("/api/compare")
def compare():
    d = request.get_json(force=True)
    g, s, t = to_grid(d)
    return jsonify([run(a, g, s, t) for a in ALGORITHMS])


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        raise ValueError("no file uploaded")
    suffix = ".map" if f.filename.endswith(".map") else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
    try:
        g = load_any(tmp.name)
    except Exception:
        raise ValueError(
            "could not parse file - expected a Moving AI .map or a 0/1 text grid"
        )
    s, t = default_endpoints(g)
    return jsonify(
        {
            "rows": g.shape[0],
            "cols": g.shape[1],
            "grid": rows_of(g),
            "start": list(s),
            "goal": list(t),
        }
    )


@app.get("/api/results")
def results():
    res = ROOT / "results"
    out = {}
    for key, fn in (
        ("windows", "summary_windows_all_algorithms.csv"),
        ("full", "summary_full_maps.csv"),
    ):
        f = res / fn
        out[key] = list(csv.DictReader(open(f))) if f.exists() else []
    fig_dir = res / "figures"
    out["figures"] = (
        sorted(p.name for p in fig_dir.glob("*")) if fig_dir.is_dir() else []
    )
    out["files"] = sorted(
        p.name for p in res.glob("*") if p.is_file() and p.suffix in (".csv", ".json")
    )
    return jsonify(out)


@app.get("/results/<path:name>")
def results_file(name):
    return send_from_directory(ROOT / "results", name)


if __name__ == "__main__":
    print(f"Map folder: {DATA_DIR}  ({len(INDEX)} maps)")
    app.run(host="0.0.0.0", port=5000, debug=False)
