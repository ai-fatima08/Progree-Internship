import os
import random
from collections import deque
from pathlib import Path

import numpy as np

PASSABLE = ".GS"
CATEGORIES = {
    "maze": "Mazes (512x512)",
    "random": "Random obstacles (512x512)",
    "dao": "Dragon Age: Origins",
    "da2": "Dragon Age 2",
}
MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


def default_data_dir():
    env = os.environ.get("MAP_DATA_DIR")
    return (
        Path(env)
        if env
        else Path(__file__).resolve().parent.parent / "data" / "movingai"
    )


# loading
def load_moving_ai(path):
    """Parse a Moving AI .map file into a uint8 grid (0 = free, 1 = wall)."""
    with open(path, errors="ignore") as fh:
        lines = fh.read().splitlines()
    h = int(next(l for l in lines if l.startswith("height")).split()[1])
    w = int(next(l for l in lines if l.startswith("width")).split()[1])
    body = lines[lines.index("map") + 1 :][:h]
    g = np.ones((h, w), dtype=np.uint8)
    for r, row in enumerate(body):
        arr = np.frombuffer(row[:w].encode("latin-1", "ignore"), dtype=np.uint8)
        g[r, : len(arr)] = ~np.isin(
            arr, np.frombuffer(PASSABLE.encode(), dtype=np.uint8)
        )
    return g


def load_txt(path):
    """Plain space-separated 0/1 grid (1 = wall)."""
    with open(path) as fh:
        rows = [[int(x) for x in line.split()] for line in fh if line.strip()]
    return np.array(rows, dtype=np.uint8)


def load_any(path):
    return load_moving_ai(path) if str(path).endswith(".map") else load_txt(path)


def _header(path):
    h = w = None
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("height"):
                h = int(line.split()[1])
            elif line.startswith("width"):
                w = int(line.split()[1])
            elif line.strip() == "map":
                break
    return h, w


def index_maps(root):
    """List every .map file below root as {name, category, path, height, width}."""
    rows = []
    root = Path(root)
    for cat in CATEGORIES:
        d = root / cat
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.map")):
            h, w = _header(f)
            rows.append(
                {
                    "name": f.stem,
                    "category": cat,
                    "path": str(f),
                    "height": h,
                    "width": w,
                }
            )
    return rows


def map_param(name):
    """maze512-8-3 -> (8, 3); random512-25-1 -> (25, 1); other maps -> (None, None)."""
    parts = name.split("-")
    if len(parts) == 3 and parts[0] in ("maze512", "random512"):
        return int(parts[1]), int(parts[2])
    return None, None


#  connectivity 
def bfs_dist(grid, src):
    """Distance map (-1 = unreachable) from src with 4-connected moves."""
    R, C = grid.shape
    cells = grid.tolist()
    dist = [[-1] * C for _ in range(R)]
    dist[src[0]][src[1]] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        d = dist[r][c] + 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not cells[nr][nc] and dist[nr][nc] < 0:
                dist[nr][nc] = d
                q.append((nr, nc))
    return np.array(dist, dtype=np.int32)


def largest_component(grid):
    """Boolean mask of the biggest 4-connected free region (single-pass labelling, O(cells))."""
    R, C = grid.shape
    cells = grid.tolist()
    label = [[0] * C for _ in range(R)]
    sizes = [0]
    for r0 in range(R):
        for c0 in range(C):
            if cells[r0][c0] or label[r0][c0]:
                continue
            lid = len(sizes)
            label[r0][c0] = lid
            q = deque([(r0, c0)])
            n = 0
            while q:
                r, c = q.popleft()
                n += 1
                for dr, dc in MOVES:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < R
                        and 0 <= nc < C
                        and not cells[nr][nc]
                        and not label[nr][nc]
                    ):
                        label[nr][nc] = lid
                        q.append((nr, nc))
            sizes.append(n)
    if len(sizes) == 1:
        return np.zeros(grid.shape, dtype=bool)
    best = max(range(1, len(sizes)), key=lambda k: sizes[k])
    return np.array(label) == best


def make_pair(grid, rng, comp=None, far=0.7):
    """Start/goal inside the largest component; goal is at BFS distance >= far * max distance."""
    comp = largest_component(grid) if comp is None else comp
    cells = np.argwhere(comp)
    if len(cells) < 2:
        raise ValueError("map has no connected free region")
    s = tuple(int(x) for x in cells[rng.randrange(len(cells))])
    d = bfs_dist(grid, s)
    mx = int(d.max())
    cand = np.argwhere(d >= int(mx * far))
    t = tuple(int(x) for x in cand[rng.randrange(len(cand))])
    return s, t, int(d[t])


def make_scenarios(grid, n=2, seed=0):
    """n reproducible (start, goal, bfs_distance) triples for a map."""
    rng = random.Random(seed)
    comp = largest_component(grid)
    out = []
    for _ in range(n):
        out.append(make_pair(grid, rng, comp))
    return out


def best_window(grid, size, rng, tries=60):
    """Pick a size x size window with many free cells (for Q-Learning). Returns (r0, c0)."""
    R, C = grid.shape
    size = min(size, R, C)
    ii = np.pad((grid == 0).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_free = (0, 0), -1
    for _ in range(tries):
        r0, c0 = rng.randrange(R - size + 1), rng.randrange(C - size + 1)
        free = (
            ii[r0 + size, c0 + size]
            - ii[r0, c0 + size]
            - ii[r0 + size, c0]
            + ii[r0, c0]
        )
        if 0.25 * size * size <= free <= 0.9 * size * size and free > best_free:
            best, best_free = (r0, c0), free
    if best_free < 0:  # nothing in the preferred band: take the fullest window
        for _ in range(tries):
            r0, c0 = rng.randrange(R - size + 1), rng.randrange(C - size + 1)
            free = (
                ii[r0 + size, c0 + size]
                - ii[r0, c0 + size]
                - ii[r0 + size, c0]
                + ii[r0, c0]
            )
            if free > best_free:
                best, best_free = (r0, c0), free
    return best


def window_scenario(grid, size=40, seed=0):
    """Crop a window and give it a start/goal pair: (sub_grid, (r0, c0), start, goal, bfs_distance)."""
    rng = random.Random(seed)
    r0, c0 = best_window(grid, size, rng)
    sub = grid[r0 : r0 + size, c0 : c0 + size].copy()
    s, t, dist = make_pair(sub, rng, far=0.6)
    return sub, (r0, c0), s, t, dist


def default_endpoints(grid):
    """Deterministic fallback for uploaded grids: first / last free cell of the largest component."""
    comp = largest_component(grid)
    cells = np.argwhere(comp)
    if len(cells) < 2:
        cells = np.argwhere(grid == 0)
    return tuple(int(x) for x in cells[0]), tuple(int(x) for x in cells[-1])
