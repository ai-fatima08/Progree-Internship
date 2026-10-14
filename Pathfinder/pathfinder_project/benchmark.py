"""Benchmark A*, Dijkstra and Q-Learning on the Moving AI grid maps and export logs + figures.

    python benchmark.py                       # data/movingai (or $MAP_DATA_DIR)
    python benchmark.py --quick               # small smoke run
    python benchmark.py --ids 0 1 2 --scen 3  # more maps / scenarios per map

Two kinds of test
  full    : the entire map (512x512 for maze/random, up to ~800x800 for game maps), A* and Dijkstra.
            The shortest-path length is cross-checked against an independent BFS.
  window  : a 40x40 crop of the map, all three algorithms (tabular Q-Learning cannot handle a 262k-state map).
Outputs: results/*.csv, results/*.json, results/figures/*.png|gif
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from pathfinder.algorithms import ALGORITHMS
from pathfinder.dataset import (
    default_data_dir,
    index_maps,
    load_moving_ai,
    make_scenarios,
    map_param,
    window_scenario,
)

OUT = Path(__file__).parent / "results"
FIG = OUT / "figures"
COL = {"A*": "#0f9d8a", "Dijkstra": "#6d4aff", "Q-Learning": "#e8590c"}
CAT_LABEL = {
    "maze": "Maze 512",
    "random": "Random 512",
    "dao": "Dragon Age: Origins",
    "da2": "Dragon Age 2",
}
WINDOW = 40


#  task workers
def _row(meta, grid, res, optimal):
    return {
        **meta,
        "algorithm": res["algorithm"],
        "found": res["found"],
        "steps": res["path_len"],
        "optimal_steps": optimal,
        "gap": (
            round(res["path_len"] / optimal - 1, 4)
            if res["found"] and optimal
            else None
        ),
        "runtime_ms": round(res["runtime_ms"], 3),
        "expanded_nodes": res["expanded_nodes"],
        "free_cells": res["free_cells"],
        "expanded_density": round(res["expanded_density"], 4),
        "episodes": res.get("episodes_run"),
        "training_steps": res.get("training_steps"),
    }


def run_full(task):
    """task = (map record, scenario index, seed). Runs A* and Dijkstra on the whole map."""
    rec, k, nscen = task
    grid = load_moving_ai(rec["path"])
    s, t, bfs = make_scenarios(grid, nscen, seed=7)[k]
    p, i = map_param(rec["name"])
    meta = {
        "category": rec["category"],
        "map": rec["name"],
        "param": p,
        "scope": "full",
        "scenario": k,
        "rows": grid.shape[0],
        "cols": grid.shape[1],
        "wall_pct": round(float(grid.mean()) * 100, 2),
        "start": f"{s[0]},{s[1]}",
        "goal": f"{t[0]},{t[1]}",
        "bfs_dist": bfs,
    }
    out = []
    for a in ("A*", "Dijkstra"):
        res = ALGORITHMS[a](grid, s, t)
        row = _row(meta, grid, res, bfs)
        row["verified_optimal"] = res["path_len"] == bfs
        out.append(row)
    return out


def run_window(task):
    """task = (map record, q_time_limit). Crops a 40x40 window and runs all three algorithms."""
    rec, qlim = task
    grid = load_moving_ai(rec["path"])
    sub, (r0, c0), s, t, bfs = window_scenario(grid, WINDOW, seed=11)
    p, _ = map_param(rec["name"])
    meta = {
        "category": rec["category"],
        "map": rec["name"],
        "param": p,
        "scope": "window",
        "scenario": 0,
        "rows": sub.shape[0],
        "cols": sub.shape[1],
        "wall_pct": round(float(sub.mean()) * 100, 2),
        "start": f"{s[0]},{s[1]}",
        "goal": f"{t[0]},{t[1]}",
        "bfs_dist": bfs,
        "window_origin": f"{r0},{c0}",
    }
    out = []
    for a in ("A*", "Dijkstra", "Q-Learning"):
        kw = {"time_limit": qlim} if a == "Q-Learning" else {}
        res = ALGORITHMS[a](sub, s, t, **kw)
        row = _row(meta, sub, res, bfs)
        row["verified_optimal"] = res["path_len"] == bfs
        out.append(row)
    return out


#  selection
def select_maps(index, ids, per_game):
    maps = []
    for cat, key in (("random", "random512"), ("maze", "maze512")):
        for rec in index:
            if rec["category"] == cat:
                p, i = map_param(rec["name"])
                if i in ids:
                    maps.append(rec)
    for cat in ("dao", "da2"):
        pool = sorted(
            [r for r in index if r["category"] == cat],
            key=lambda r: r["height"] * r["width"],
        )
        pool = [
            r for r in pool if r["height"] * r["width"] >= 20000
        ]  # skip tiny test maps
        step = max(1, len(pool) // per_game)
        maps += pool[::step][:per_game]
    return maps


# figures
def draw(ax, grid, res, title, big=False):
    ax.imshow(
        grid, cmap=ListedColormap(["#fbfaff", "#2b2160"]), interpolation="nearest"
    )
    if res["order"]:
        ex = np.full(grid.shape, np.nan)
        n = len(res["order"])
        for k, (r, c) in enumerate(res["order"]):
            ex[r, c] = k / n
        ax.imshow(ex, cmap="cool", alpha=0.6, interpolation="nearest", vmin=0, vmax=1)
    if res["path"]:
        ys, xs = zip(*res["path"])
        ax.plot(xs, ys, color="#ffb703", lw=0.9 if big else 2.2, solid_capstyle="round")
        sz = 30 if big else 60
        ax.scatter([xs[0]], [ys[0]], c="#2de2c8", s=sz, edgecolor="k", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], c="#ff6b7a", s=sz, edgecolor="k", zorder=5)
    ax.set_title(
        f"{title}\nsteps={res['path_len']}  expanded={res['expanded_nodes']}  {res['runtime_ms']:.1f} ms",
        fontsize=9,
    )
    ax.axis("off")


def fig_scenario(rec, scope, name, algos):
    grid = load_moving_ai(rec["path"])
    if scope == "full":
        s, t, _ = make_scenarios(grid, 1, seed=7)[0]
        g = grid
    else:
        g, _, s, t, _ = window_scenario(grid, WINDOW, seed=11)
    fig, axes = plt.subplots(1, len(algos), figsize=(4.6 * len(algos), 5))
    axes = np.atleast_1d(axes)
    results = {}
    for ax, a in zip(axes, algos):
        kw = {"record_paths": True} if a == "Q-Learning" else {}
        res = ALGORITHMS[a](g, s, t, **kw)
        results[a] = res
        draw(ax, g, res, a, big=scope == "full")
    fig.suptitle(
        f"{rec['name']} ({scope}{' 40x40 window' if scope == 'window' else ''}) - cyan to pink = expansion order, gold = final path",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / f"paths_{name}.png", dpi=120)
    plt.close(fig)
    return g, results


def fig_gif(grid, res, name):
    frames = [c for c in res["curve"] if "path" in c]
    if len(frames) < 2:
        return
    frames = frames[:: max(1, len(frames) // 40)] + [frames[-1]]
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(5, 5.3))

    def upd(i):
        ax.clear()
        ax.imshow(
            grid, cmap=ListedColormap(["#fbfaff", "#2b2160"]), interpolation="nearest"
        )
        ys, xs = zip(*frames[i]["path"])
        ax.plot(xs, ys, color="#e8590c", lw=2.4)
        ax.set_title(
            f"Q-Learning path optimisation\nepisode {frames[i]['episode']}  path = {frames[i]['greedy_path_len']} steps"
        )
        ax.axis("off")

    FuncAnimation(fig, upd, frames=len(frames)).save(
        FIG / f"qlearning_{name}.gif", writer=PillowWriter(fps=4)
    )
    plt.close(fig)


def fig_learning_curve(items):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, (eps, opt) in items.items():
        x = [i + 1 for i, (_, ok) in enumerate(eps) if ok]
        y = [st for st, ok in eps if ok]
        if not x:
            continue
        k = max(1, len(y) // 12)
        ys = [
            sum(y[max(0, i - k) : i + 1]) / len(y[max(0, i - k) : i + 1])
            for i in range(len(y))
        ]
        (line,) = ax.plot(x, ys, lw=1.8, label=f"{label} (optimal {opt})")
        ax.axhline(opt, ls=":", lw=1, color=line.get_color())
    ax.set_yscale("log")
    ax.set(
        xlabel="training episode",
        ylabel="steps agent needed to reach goal (log)",
        title="Q-Learning path optimisation: steps per episode fall toward the A* optimum",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "qlearning_learning_curve.png", dpi=130)
    plt.close(fig)


def fig_param_sweep(df, cat, xlabel, fname, title):
    d = df[(df.category == cat) & (df.scope == "full")]
    if d.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, g in d.groupby("algorithm"):
        m = (
            g.groupby("param")
            .agg(
                wall=("wall_pct", "mean"),
                dens=("expanded_density", "mean"),
                rt=("runtime_ms", "mean"),
                st=("steps", "mean"),
            )
            .reset_index()
        )
        x = m["wall"] if cat == "random" else m["param"]
        ax[0].plot(x, m["dens"] * 100, marker="o", color=COL[a], label=a)
        ax[1].plot(x, m["rt"], marker="o", color=COL[a], label=a)
        ax[2].plot(x, m["st"], marker="o", color=COL[a], label=a)
    for a_, yl in zip(
        ax, ["expanded density (% of free cells)", "runtime (ms)", "path steps"]
    ):
        a_.set(xlabel=xlabel, ylabel=yl)
        a_.grid(alpha=0.3)
        a_.legend()
    if cat == "maze":
        for a_ in ax:
            a_.set_xscale("log", base=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def fig_summary(df):
    f = df[df.scope == "full"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for a, g in f.groupby("algorithm"):
        ax.scatter(
            g["free_cells"], g["runtime_ms"], s=18, alpha=0.7, color=COL[a], label=a
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="free cells in map",
        ylabel="runtime (ms)",
        title="Runtime vs map size (full maps)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "runtime_vs_free_cells.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    m = f.groupby(["category", "algorithm"])["expanded_density"].mean().unstack() * 100
    m = m.reindex([c for c in CAT_LABEL if c in m.index])
    m.index = [CAT_LABEL[c] for c in m.index]
    m.plot.bar(ax=ax, color=[COL[c] for c in m.columns], rot=0)
    ax.set_ylabel("expanded density (% of free cells)")
    ax.set_title("Share of free cells each algorithm expanded (full maps)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "expanded_density_by_category.png", dpi=130)
    plt.close(fig)


# main
def agg(d):
    return (
        d.groupby(["category", "algorithm"])
        .agg(
            runs=("map", "count"),
            solved=("found", "sum"),
            mean_steps=("steps", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
            mean_expanded=("expanded_nodes", "mean"),
            mean_expanded_density=("expanded_density", "mean"),
            mean_gap=("gap", "mean"),
        )
        .round(4)
        .reset_index()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(default_data_dir()))
    ap.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[0, 1],
        help="map ids used for maze512 / random512",
    )
    ap.add_argument(
        "--scen", type=int, default=2, help="start/goal pairs per map (full-map tests)"
    )
    ap.add_argument("--game-maps", type=int, default=8, help="maps per Dragon Age set")
    ap.add_argument("--q-time-limit", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.ids, a.scen, a.game_maps, a.q_time_limit = [0], 1, 2, 15.0
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    index = index_maps(a.data)
    assert index, f"no .map files found under {a.data}"
    maps = select_maps(index, a.ids, a.game_maps)
    full_tasks = [(m, k, a.scen) for m in maps for k in range(a.scen)]
    win_tasks = [(m, a.q_time_limit) for m in maps]
    print(
        f"{len(index)} maps indexed; testing {len(maps)} maps: {len(full_tasks)} full runs x2 algorithms, "
        f"{len(win_tasks)} {WINDOW}x{WINDOW} windows x3 algorithms",
        flush=True,
    )
    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for kind, fn, tasks in (
            ("window", run_window, win_tasks),
            ("full", run_full, full_tasks),
        ):
            for i, r in enumerate(ex.map(fn, tasks), 1):
                rows += r
                if i % 5 == 0 or i == len(tasks):
                    print(
                        f"  {kind}: {i}/{len(tasks)} ({time.time()-t0:.0f}s)",
                        flush=True,
                    )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "benchmark_log.csv", index=False)
    (OUT / "benchmark_log.json").write_text(json.dumps(rows, indent=1, default=str))
    bad = df[(df.algorithm != "Q-Learning") & (~df.verified_optimal.astype(bool))]
    print(
        f"A*/Dijkstra paths equal to independent BFS optimum: {len(df[df.algorithm != 'Q-Learning']) - len(bad)}"
        f"/{len(df[df.algorithm != 'Q-Learning'])}"
    )
    full, win = df[df.scope == "full"], df[df.scope == "window"]
    agg(full).to_csv(OUT / "summary_full_maps.csv", index=False)
    agg(win).to_csv(OUT / "summary_windows_all_algorithms.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "full": agg(full).to_dict("records"),
                "windows": agg(win).to_dict("records"),
            },
            indent=1,
        )
    )
    print(
        "\nFULL MAPS\n",
        agg(full).to_string(index=False),
        "\n\n40x40 WINDOWS (same cells, all algorithms)\n",
        agg(win).to_string(index=False),
        flush=True,
    )

    fig_summary(df)
    fig_param_sweep(
        df,
        "random",
        "measured obstacle % (walls + trees) of the map",
        "obstacle_density_sweep.png",
        "Random 512x512 maps: effect of obstacle density (full maps)",
    )
    fig_param_sweep(
        df,
        "maze",
        "corridor width (cells)",
        "maze_corridor_width.png",
        "Maze 512x512 maps: effect of corridor width (full maps)",
    )
    # Path visualisations
    by_name = {m["name"]: m for m in maps}
    pick = {}
    for m in maps:
        pick.setdefault(m["category"], m)
    pick["random"] = by_name.get("random512-20-0", pick.get("random"))
    pick["maze"] = by_name.get("maze512-4-0", pick.get("maze"))
    curves, gifs = {}, 0
    for cat, rec in pick.items():
        if rec is None:
            continue
        g, res = fig_scenario(
            rec, "window", f"{cat}_window", ["A*", "Dijkstra", "Q-Learning"]
        )
        q = res["Q-Learning"]
        opt = res["A*"]["path_len"]
        if q["found"]:
            curves[f"{cat}: {rec['name']}"] = (q["episode_steps"], opt)
            if gifs < 2:
                fig_gif(g, q, cat)
                gifs += 1
        fig_scenario(rec, "full", f"{cat}_full", ["A*", "Dijkstra"])
    if curves:
        fig_learning_curve(curves)
    print(f"done in {time.time()-t0:.0f}s ->", OUT, flush=True)


if __name__ == "__main__":
    main()
