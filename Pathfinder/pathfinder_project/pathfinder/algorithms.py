"""
Grid convention: numpy 2-D array, 0 = free cell, 1 = wall. Moves are 4-connected, cost 1 each.
Every solver returns the same metrics dict so results are directly comparable:
    algorithm, found, path, path_len (steps), order (expansion order), expanded_nodes,
    expanded_density (expanded / free cells), runtime_ms, free_cells
"""

import heapq
import random
import time

import numpy as np

MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))


# helpers
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _result(name, grid, path, order, t0, found, **extra):
    free = int((grid == 0).sum())
    out = {
        "algorithm": name,
        "found": bool(found),
        "path": path if found else [],
        "path_len": len(path) - 1 if found else 0,
        "order": order,
        "expanded_nodes": len(order),
        "expanded_density": len(order) / free if free else 0.0,
        "runtime_ms": (time.perf_counter() - t0) * 1000.0,
        "free_cells": free,
    }
    out.update(extra)
    return out


def _rebuild(parent, start, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path if path[0] == start else []


#  A* / Dijkstra
def _best_first(grid, start, goal, use_heuristic, name):
    """Shared priority-queue search. Dijkstra = f=g, A* = f=g+h (Manhattan, admissible)."""
    t0 = time.perf_counter()
    R, C = grid.shape
    cells = grid.tolist()
    h = (
        (lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
        if use_heuristic
        else (lambda p: 0)
    )
    g_cost = {start: 0}
    parent = {start: None}
    closed = set()
    order = []
    heap = [(h(start), 0, start)]
    tie = 0
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            return _result(name, grid, _rebuild(parent, start, goal), order, t0, True)
        r, c = cur
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not cells[nr][nc]:
                nxt = (nr, nc)
                ng = g_cost[cur] + 1
                if nxt not in closed and ng < g_cost.get(nxt, 1 << 60):
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    tie += 1
                    heapq.heappush(heap, (ng + h(nxt), tie, nxt))
    return _result(name, grid, [], order, t0, False)


def astar(grid, start, goal):
    return _best_first(grid, start, goal, True, "A*")


def dijkstra(grid, start, goal):
    return _best_first(grid, start, goal, False, "Dijkstra")


#  Q-Learning
def q_learning(
    grid,
    start,
    goal,
    episodes=4000,
    alpha=0.5,
    gamma=0.99,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay_episodes=250,
    shaping=False,
    seed=0,
    time_limit=20.0,
    eval_every=10,
    patience=6,
    record_paths=False,
):
    """Tabular Q-Learning agent.
    State = free cell, actions = 4 moves. Reward: +100 goal, -1 per step, -5 wall bump. Because
    unvisited actions start at Q=0 (better than any penalised action) the agent explores
    systematically. Optional potential-based shaping (phi = -Manhattan, `shaping=True`) is OFF by
    default: in mazes it pulls the agent into dead ends and prevented convergence in our tests.
    Epsilon-greedy with linear decay over `eps_decay_episodes`. Every `eval_every` episodes the greedy
    policy is rolled out; training stops early once the greedy path to the goal stays unchanged
    for `patience` evaluations (convergence) or `time_limit` seconds are spent.
    The learning curve (episode -> greedy path length) is returned for the optimisation log.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    R, C = grid.shape
    free = [(int(r), int(c)) for r, c in zip(*np.where(grid == 0))]
    idx = {p: i for i, p in enumerate(free)}
    n = len(free)
    nxt = [[0] * 4 for _ in range(n)]  # next-state table, bump -> stay in place
    bump = [[False] * 4 for _ in range(n)]
    for i, (r, c) in enumerate(free):
        for a, (dr, dc) in enumerate(MOVES):
            q = (r + dr, c + dc)
            if q in idx:
                nxt[i][a] = idx[q]
            else:
                nxt[i][a] = i
                bump[i][a] = True
    phi = [-manhattan(p, goal) for p in free]
    s0, sg = idx[start], idx[goal]
    Q = [[0.0] * 4 for _ in range(n)]
    visited = {s0}
    order = [start]
    max_steps = max(200, 4 * n)

    def greedy_path():
        path, cur, seen = [s0], s0, {s0}
        while cur != sg and len(path) <= n:
            row = Q[cur]
            a = row.index(max(row))
            cur = nxt[cur][a]
            if cur in seen:
                return None
            seen.add(cur)
            path.append(cur)
        return path if cur == sg else None

    curve, stable, last_len, total_steps, ep = [], 0, None, 0, 0
    ep_steps = []  # (steps used, reached goal) per episode
    best_path = None
    for ep in range(1, episodes + 1):
        eps = max(eps_min, eps_start - (eps_start - eps_min) * ep / eps_decay_episodes)
        s = s0
        used = 0
        for _ in range(max_steps):
            used += 1
            row = Q[s]
            a = rng.randrange(4) if rng.random() < eps else row.index(max(row))
            s2 = nxt[s][a]
            reward = 100.0 if s2 == sg else (-5.0 if bump[s][a] else -1.0)
            if shaping:
                reward += gamma * phi[s2] - phi[s]
            target = reward if s2 == sg else reward + gamma * max(Q[s2])
            row[a] += alpha * (target - row[a])
            total_steps += 1
            if s2 not in visited:
                visited.add(s2)
                order.append(free[s2])
            s = s2
            if s == sg:
                break
        ep_steps.append((used, s == sg))
        if ep % eval_every == 0:
            gp = greedy_path()
            L = len(gp) - 1 if gp else None
            entry = {"episode": ep, "greedy_path_len": L, "epsilon": round(eps, 3)}
            if record_paths and gp:
                entry["path"] = [free[i] for i in gp]
            curve.append(entry)
            if gp:
                best_path = gp
                stable = stable + 1 if L == last_len else 0
                last_len = L
                if stable >= patience:
                    break
            else:
                stable, last_len = 0, None
            if time.perf_counter() - t0 > time_limit:
                break
    gp = greedy_path() or best_path
    path = [free[i] for i in gp] if gp else []
    return _result(
        "Q-Learning",
        grid,
        path,
        order,
        t0,
        bool(gp),
        episodes_run=ep,
        training_steps=total_steps,
        curve=curve,
        episode_steps=ep_steps,
    )


ALGORITHMS = {"A*": astar, "Dijkstra": dijkstra, "Q-Learning": q_learning}
