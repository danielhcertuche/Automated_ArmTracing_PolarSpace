#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm‑Tracing Pipeline (consolidated single script)
-------------------------------------------------
This script unifies the four notebook sections the user provided into a single,
self‑contained module while preserving every original function and the overall
algorithmic flow.  The *logic* and numerical outputs remain **unchanged**; the
additions focus strictly on:

*   Lightweight `print()` statements that announce when key stages complete.
*   Optional saving of every figure produced (default enabled).
*   Minimal refactor for readability and best practices—e.g. main entry point,
    configurable I/O paths, and type hints—without touching the core maths.

Usage
~~~~~
Run a single Subhalo ID (default 418336) ::

    python arm_tracing_pipeline.py 418336 --out figures

or multiple IDs ::

    python arm_tracing_pipeline.py 418336 117251 372755 --out figs_all

Every plot is saved as PNG in the chosen folder and still shown interactively
when an attached display is available.
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import heapq
from scipy.ndimage import binary_closing, gaussian_filter, label
from sklearn.linear_model import LinearRegression
from matplotlib import cm

# -----------------------------------------------------------------------------
# 0 – UTILITIES
# -----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
})


def _ensure_dir(path: str) -> None:
    """Create *path* and parents if they do not yet exist."""
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# 1 – PITCH‑ANGLE
# -----------------------------------------------------------------------------

def calculate_pa(slope: float, intercept: float) -> float:
    """Return pitch angle in degrees; *NaN* if *intercept* is zero."""
    if intercept != 0:
        return np.degrees(np.arctan((slope * (180 / np.pi)) / intercept))
    return float("nan")


# -----------------------------------------------------------------------------
# 2 – DATA IO
# -----------------------------------------------------------------------------

def load_and_filter_data(
    id_halo: str | int,
    theta_min: float,
    theta_max: float,
    *,
    file_prefix: str = "data_rho",
) -> pd.DataFrame:
    """Read *{file_prefix}_{id}.csv*, compute **r**, **theta**, and filter range."""
    df = pd.read_csv(f"{file_prefix}_{id_halo}_filtered.csv")
    df["id"] = df.index  # preserve original index
    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    #df["theta"] = np.degrees(np.arctan2(df["y"], df["x"]))
    #df.loc[df["theta"] < 0, "theta"] += 360

    base_deg = np.degrees(np.arctan2(df["y"], df["x"]))
    df["theta1"] = base_deg
    df["theta2"] = base_deg + 360.0

    # Creo dos copias del DataFrame, una para cada theta
    df1 = df.copy()
    df1["theta"] = df1["theta1"]
    df2 = df.copy()
    df2["theta"] = df2["theta2"]

    # Concateno manteniendo todas las columnas
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # Filtro según theta unificado
    df_filtered = df_combined[
        (df_combined["theta"] >= theta_min) &
        (df_combined["theta"] <= theta_max)
    ].copy()

    # Aseguro tipos float en las columnas numéricas relevantes
    df_filtered["r"]     = df_filtered["r"].astype(float)
    df_filtered["theta"] = df_filtered["theta"].astype(float)

    print(
        f"Loaded {len(df_filtered):,} pts for halo {id_halo} | "
        f"θ ∈ [{theta_min}, {theta_max}]° (combined)"
    )
    return df_filtered


# -----------------------------------------------------------------------------
# 3 – GRAPH BUILDING
# -----------------------------------------------------------------------------

def build_graph_rectangular(
    df_points: pd.DataFrame, theta_diff: float, r_diff: float
) -> Tuple[List[List[int]], int]:
    """Axis‑aligned proximity graph used by BFS."""
    th, rr = df_points["theta"].values, df_points["r"].values
    n = len(df_points)
    graph: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        mask = (
            (np.abs(th[i] - th) <= theta_diff)
            & (np.abs(rr[i] - rr) <= r_diff)
            & (np.arange(n) != i)
        )
        graph[i] = list(np.where(mask)[0])
    return graph, n


def bfs_components(graph: List[List[int]], df_points: pd.DataFrame):
    """Return list of DataFrames, each a connected component via BFS."""
    visited = [False] * len(graph)
    clusters: List[pd.DataFrame] = []
    for s in range(len(graph)):
        if not visited[s]:
            q, comp = deque([s]), [s]
            visited[s] = True
            while q:
                u = q.popleft()
                for v in graph[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
                        comp.append(v)
            clusters.append(df_points.iloc[comp].copy())
    return clusters


# -----------------------------------------------------------------------------
# 4 – GAP‑BASED SUBDIVISION
# -----------------------------------------------------------------------------

def subdivide_by_gap(
    df_cluster: pd.DataFrame, gap_threshold: float = 2.5, mode: str = "theta"
):
    col = "theta" if mode == "theta" else "r"
    df = df_cluster.sort_values(col)
    dif = np.diff(df[col].values)
    if np.all(dif <= gap_threshold):
        return [df]
    subs, start = [], 0
    for i, d in enumerate(dif):
        if d > gap_threshold:
            subs.append(df.iloc[start : i + 1].copy())
            start = i + 1
    subs.append(df.iloc[start:].copy())
    return subs



# -----------------------------------------------------------------------------
# 5 – SEED GENERATION
# -----------------------------------------------------------------------------

def _adaptive_factor(n_pts: int, *, ref: float = 2000) -> float:
    return math.sqrt(max(n_pts / ref, 1.0))


def generate_bfs_seeds(
    *,
    id_halo: str | int,
    theta_min: float = 30,
    theta_max: float = 540,
    quartile_threshold: float = 0.10,
    theta_diff: float = 3.0,
    r_diff: float = 0.5,
    gap_threshold_theta: float = 2.0,
    gap_threshold_r: float = 2.0,
    file_prefix: str = "data_rho",
):
    """Return BFS‑based clusters above a quantile cutoff plus the filtered DF."""
    df = load_and_filter_data(id_halo, theta_min, theta_max, file_prefix=file_prefix)
    if len(df) > 20000:
        used_q = 0.25
        print(f"⚠️ Advertencia: {len(df)} partículas > 20000, usando cuantil 25% en lugar de {quartile_threshold*100:.1f}%")

    else:
        used_q = quartile_threshold
        print(f"Usando cuantil {used_q*100:.1f}% (quartile_threshold originalmente {quartile_threshold*100:.1f}%)")


    thr = df["rho_resta_final_exp"].quantile(used_q)
    df_f = df[df["rho_resta_final_exp"] > thr].copy().reset_index(drop=True)

    f = _adaptive_factor(len(df_f))
    th_diff, r_d = theta_diff / f, r_diff / f
    gap_th, gap_r = gap_threshold_theta / f, gap_threshold_r / f

    graph, _ = build_graph_rectangular(df_f, th_diff, r_d)
    clusters = bfs_components(graph, df_f)

    final_cl: List[pd.DataFrame] = []
    for cl in clusters:
        for st in subdivide_by_gap(cl, gap_th, "theta"):
            final_cl.extend(subdivide_by_gap(st, gap_r, "r"))

    print(f"Generated {len(final_cl)} BFS seed clusters (>{quartile_threshold*100:.0f}%-ile).")

    #graficas
    plt.figure()

    # Histograma de todo el dataset
    plt.hist(
        df["rho_resta_final_exp"],
        bins=50,
        alpha=0.56,
        label="Datos originales (df)"
    )

    # Histograma de los datos filtrados
    plt.hist(
        df_f["rho_resta_final_exp"],
        bins=50,
        alpha=0.56,
        label=f"Datos retenidos (df_f, q={used_q*100:.0f}%)"
    )

    # Línea vertical en el valor de corte
    plt.axvline(
        thr,
        linestyle="--",
        label=f"Umbral = {thr:.3f}"
    )

    # Configurar etiquetas y título
    plt.xlabel("rho_resta_final_exp")
    plt.ylabel("Frecuencia")
    plt.title("Selección por cuantil: totalidad vs filtrado")
    plt.legend()
    plt.tight_layout()

    # Guardar la figura en disco
    plt.savefig("selection_plot.png", dpi=300)
    plt.close()

    return final_cl, df_f


# -----------------------------------------------------------------------------
# 6 – LINE FIT & GROUP MERGE / VALIDATE
# -----------------------------------------------------------------------------

def fit_line_to_cluster(
    df_cluster: pd.DataFrame,
):
    if len(df_cluster) < 2:
        return (None,) * 7
    mdl = LinearRegression().fit(df_cluster[["theta"]], df_cluster["r"])
    s, b = mdl.coef_[0], mdl.intercept_
    pa = calculate_pa(s, b)
    return (
        s,
        b,
        pa,
        df_cluster["theta"].min(),
        df_cluster["theta"].max(),
        df_cluster["r"].min(),
        df_cluster["r"].max(),
    )


def adjust_and_merge_seeds(
    bfs_clusters: List[pd.DataFrame],
    *,
    slope_variation_threshold: float = 0.40,
    bounding_extrap: float = 0.30,
):
    """Merge overlapping clusters with consistent slope."""
    groups: List[Dict[str, Any]] = []
    for cl in bfs_clusters:
        groups.append(
            dict(
                zip(
                    [
                        "slope",
                        "intercept",
                        "pa",
                        "theta_min",
                        "theta_max",
                        "r_min",
                        "r_max",
                        "points",
                    ],
                    (*fit_line_to_cluster(cl), cl),
                )
            )
        )

    def recalc(g):
        (
            g["slope"],
            g["intercept"],
            g["pa"],
            g["theta_min"],
            g["theta_max"],
            g["r_min"],
            g["r_max"],
        ) = fit_line_to_cluster(g["points"])

    def boxes_overlap(g1: Dict[str, Any], g2: Dict[str, Any], e: float):
        def exp(t0, t1, r0, r1):
            dt, dr = (t1 - t0) * e, (r1 - r0) * e
            return t0 - dt, t1 + dt, r0 - dr, r1 + dr

        a = exp(g1["theta_min"], g1["theta_max"], g1["r_min"], g1["r_max"])
        b = exp(g2["theta_min"], g2["theta_max"], g2["r_min"], g2["r_max"])
        return not (a[1] < b[0] or b[1] < a[0]) and not (a[3] < b[2] or b[3] < a[2])

    merged = True
    while merged:
        merged, new = False, []
        i = 0
        while i < len(groups):
            g1, j = groups[i], i + 1
            while j < len(groups):
                g2 = groups[j]
                if g1["slope"] is None or g2["slope"] is None:
                    j += 1
                    continue
                if boxes_overlap(g1, g2, bounding_extrap):
                    comb = pd.concat([g1["points"], g2["points"]], ignore_index=True)
                    s_comb, *_ = fit_line_to_cluster(comb)
                    if s_comb is not None:
                        v1 = abs(s_comb - g1["slope"]) / (abs(g1["slope"]) + 1e-12)
                        v2 = abs(s_comb - g2["slope"]) / (abs(g2["slope"]) + 1e-12)
                        if v1 < slope_variation_threshold and v2 < slope_variation_threshold:
                            g1["points"] = comb
                            recalc(g1)
                            groups.pop(j)
                            merged = True
                            continue
                j += 1
            new.append(g1)
            i += 1
        groups = new

    print(f"Merged groups: {len(groups)}")
    return [g for g in groups if len(g["points"]) >= 2]


def validate_dispersion_and_reprocess(
    groups: List[Dict[str, Any]],
    *,
    dispersion_threshold: float = 1.80,
    reproc_theta_diff: float = 1.25,
    reproc_r_diff: float = 0.4,
):
    """Split groups whose residual dispersion exceeds *dispersion_threshold*."""
    ok, todo = [], []
    for g in groups:
        if g["slope"] is None or len(g["points"]) < 2:
            ok.append(g)
            continue
        res = g["points"]["r"] - (g["slope"] * g["points"]["theta"] + g["intercept"])
        (todo if res.std() > dispersion_threshold else ok).append(g)

    new: List[Dict[str, Any]] = []
    for g in todo:
        subdf = g["points"].reset_index(drop=True)
        graph, _ = build_graph_rectangular(subdf, reproc_theta_diff, reproc_r_diff)
        for c in bfs_components(graph, subdf):
            new.append(
                dict(
                    zip(
                        [
                            "slope",
                            "intercept",
                            "pa",
                            "theta_min",
                            "theta_max",
                            "r_min",
                            "r_max",
                            "points",
                        ],
                        (*fit_line_to_cluster(c), c),
                    )
                )
            )

    print(f"Validated groups: kept {len(ok)}, re‑processed {len(new)} → total {len(ok)+len(new)}")
    return ok + new


# -----------------------------------------------------------------------------
# 7 – PLOTTING HELPERS (with optional saving)
# -----------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: str | None):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    fig.show()


def plot_groups_cartesian(
    groups: List[Dict[str, Any]],
    df_all: pd.DataFrame,
    *,
    line_extrap: float = 0.15,
    save_path: str | None = None,
):
    big = [g for g in groups if len(g["points"]) >= 60]
    print(f"Plotting Cartesian groups ≥60 pts ({len(big)}) …")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(df_all["theta"], df_all["r"], s=3, alpha=0.3, color="gray")
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "magenta",
        "cyan",
        "gold",
        "gray",
        "lime",
    ]
    for i, g in enumerate(big):
        c = colors[i % len(colors)]
        pts = g["points"]
        pa_txt = f"{g['pa']:.2f}°" if g["pa"] is not None else "—"
        plt.scatter(pts["theta"], pts["r"], s=10, color=c, label=f"G{i+1} ({len(pts)}) | PA={pa_txt}")
        if g["slope"] is not None:
            dt = g["theta_max"] - g["theta_min"]
            t = np.linspace(g["theta_min"] - line_extrap * dt, g["theta_max"] + line_extrap * dt, 200)
            plt.plot(t, g["slope"] * t + g["intercept"], "--", color=c)
    plt.xlabel("θ (°)")
    plt.ylabel("r")
    plt.grid(True)
    plt.title("Subhalo – grupos ≥60 pts (cartesiano, PA visible)")
    #plt.legend()
    _save_or_show(fig, save_path)


def plot_groups_polar(
    groups: List[Dict[str, Any]],
    df_all: pd.DataFrame,
    *,
    line_extrap: float = 0.15,
    save_path: str | None = None,
):
    big = [g for g in groups if len(g["points"]) >= 60]
    print(f"Plotting Polar groups ≥60 pts ({len(big)}) …")

    fig = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, projection="polar")
    ax.scatter(np.radians(df_all["theta"]), df_all["r"], s=3, alpha=0.3, color="gray")

    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "magenta",
        "cyan",
        "gold",
        "gray",
        "lime",
    ]
    for i, g in enumerate(big):
        c = colors[i % len(colors)]
        pts = g["points"]
        pa_txt = f"{g['pa']:.2f}°" if g["pa"] is not None else "—"
        ax.scatter(
            np.radians(pts["theta"]),
            pts["r"],
            s=10,
            color=c,
            label=f"G{i+1} ({len(pts)}) | PA={pa_txt}",
        )
        if g["slope"] is not None:
            dt = g["theta_max"] - g["theta_min"]
            t_deg = np.linspace(
                g["theta_min"] - line_extrap * dt, g["theta_max"] + line_extrap * dt, 200
            )
            r_line = g["slope"] * t_deg + g["intercept"]
            ax.plot(np.radians(t_deg), r_line, "--", color=c)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_title("Subhalo – proyección polar (PA visible)")
    #ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.0))
    _save_or_show(fig, save_path)


# -----------------------------------------------------------------------------
# 8 – HISTOGRAM SEGMENTER (unchanged, only minor typing + saving helper)
# -----------------------------------------------------------------------------

@dataclass
class Connection:
    a: Tuple[float, float]
    b: Tuple[float, float]
    delta_r: float
    euclidean: float


@dataclass
class IslandObject:
    type: str
    boundary: List[Tuple[float, float]]


class HistogramSegmenter:
    """Reusable 2‑D histogram island segmenter (ver original docstring)."""

    def __init__(
        self,
        bins_theta: int = 100,
        bins_r: int = 80,
        smooth_sigma: float = 1.0,
        density_percentile: float = 80,
        closing_size: int = 2,
        min_cluster_size: int = 40,
        max_gap_dist: float = 0.45,
        theta_bin_size: float = 8,
        density_ratio: float = 0.25,
        theta_step: float = 3,
        r_threshold: float = 0.9,
        dr_multiplier: float = 3.0,
    ):
        self.bins_theta = bins_theta
        self.bins_r = bins_r
        self.smooth_sigma = smooth_sigma
        self.density_percentile = density_percentile
        self.closing_size = closing_size
        self.min_cluster_size = min_cluster_size
        self.max_gap_dist = max_gap_dist
        self.theta_bin_size = theta_bin_size
        self.density_ratio = density_ratio
        self.theta_step = theta_step
        self.r_threshold = r_threshold
        self.dr_multiplier = dr_multiplier

    # ------------------------------------------------------------------
    # Paso 1 – Histograma + suavizado
    # ------------------------------------------------------------------
    def _histogram(self, pts: np.ndarray):
        hist, te, re = np.histogram2d(
            pts[:, 0], pts[:, 1],
            bins=[self.bins_theta, self.bins_r]
        )
        hist_s = gaussian_filter(hist, sigma=self.smooth_sigma)
        return hist_s, te, re

    # ------------------------------------------------------------------
    # Paso 2 – BFS ponderado para unir islas próximas
    # ------------------------------------------------------------------
    def _weighted_bfs(self, hist_s, te, re):
        thr = np.percentile(hist_s[hist_s > 0], self.density_percentile)
        mask = binary_closing(hist_s > thr,
                              structure=np.ones((self.closing_size,
                                                 self.closing_size)))
        labels, n_lab = label(mask)

        theta_res, r_res = te[1] - te[0], re[1] - re[0]
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        parent = list(range(n_lab + 1))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        h, w = labels.shape
        dist_grid = np.full((h, w), np.inf)
        pq: List[Tuple[float, int, int, int]] = []

        for i in range(h):
            for j in range(w):
                lab = labels[i, j]
                if lab:
                    for di, dj in neigh:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] == 0:
                            d0 = math.hypot(di * theta_res, dj * r_res)
                            if d0 <= self.max_gap_dist and d0 < dist_grid[ni, nj]:
                                dist_grid[ni, nj] = d0
                                heapq.heappush(pq, (d0, ni, nj, lab))

        while pq:
            dist, i, j, lab = heapq.heappop(pq)
            if dist > dist_grid[i, j] or dist > self.max_gap_dist:
                continue
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    lab2 = labels[ni, nj]
                    if lab2 and lab2 != lab:
                        union(lab, lab2)
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] == 0:
                    nd = dist + math.hypot(di * theta_res, dj * r_res)
                    if nd <= self.max_gap_dist and nd < dist_grid[ni, nj]:
                        dist_grid[ni, nj] = nd
                        heapq.heappush(pq, (nd, ni, nj, lab))
        return find, labels

    # ------------------------------------------------------------------
    # Paso 3 – Clusters y esqueletos
    # ------------------------------------------------------------------
    def _clusters_and_skeletons(self, pts, find, labels, te, re):
        tidx = np.clip(np.digitize(pts[:, 0], te) - 1, 0, self.bins_theta - 1)
        ridx = np.clip(np.digitize(pts[:, 1], re) - 1, 0, self.bins_r - 1)
        merged: Dict[int, List[np.ndarray]] = {}
        for p, (ti, ri) in enumerate(zip(tidx, ridx)):
            lab = labels[ti, ri]
            if lab:
                merged.setdefault(find(lab), []).append(pts[p])
        clusters = [np.vstack(v) for v in merged.values()
                    if len(v) >= self.min_cluster_size]

        skeletons: List[List[Tuple[float, float]]] = []
        for cl in clusters:
            thetas, rs = cl[:, 0], cl[:, 1]
            bins_arr = np.arange(thetas.min(),
                                 thetas.max() + self.theta_bin_size,
                                 self.theta_bin_size)
            ske = []
            for k in range(len(bins_arr) - 1):
                m = (thetas >= bins_arr[k]) & (thetas < bins_arr[k + 1])
                if m.any():
                    ske.append((thetas[m].mean(), rs[m].mean()))
            skeletons.append(sorted(ske, key=lambda x: -x[0]))
        return clusters, skeletons

    # ------------------------------------------------------------------
    # Paso 4 – Centroides adicionales sobre *todas* las rectas candidatas
    # ------------------------------------------------------------------
    def _density_path(self, a, b, hist_s, te, re):
        """Genera centroides extra a lo largo del segmento a‑b."""
        theta_res, r_res = te[1] - te[0], re[1] - re[0]
        steps = max(int(abs(b[0] - a[0]) / self.theta_step) + 1, 2)
        thetas = np.linspace(a[0], b[0], steps)
        rs_seg = np.linspace(a[1], b[1], steps)
        radius = math.hypot(b[0] - a[0], b[1] - a[1]) * self.density_ratio

        out: List[Tuple[float, float]] = []
        for θ_i, r_i in zip(thetas, rs_seg):
            ti = int(np.clip(np.digitize(θ_i, te) - 1, 0, hist_s.shape[0] - 1))
            ri0 = int(np.clip(np.digitize(r_i, re) - 1, 0, hist_s.shape[1] - 1))
            max_off = int(math.ceil(radius / r_res))
            vals, offs = [], []
            for off in range(-max_off, max_off + 1):
                idx = ri0 + off
                if 0 <= idx < hist_s.shape[1]:
                    vals.append(hist_s[ti, idx]); offs.append(off)
            for k in range(1, len(vals) - 1):
                if vals[k] > vals[k - 1] and vals[k] > vals[k + 1] and vals[k] > 0:
                    rc = re[ri0 + offs[k]] + r_res / 2.0
                    if abs(rc - r_i) <= self.r_threshold:
                        out.append((θ_i, rc))
        return out

    def _candidate_segments(self, skeletons):
        segs = []
        # 4a) internos (pares contiguos)
        for ske in skeletons:
            segs.extend(list(zip(ske, ske[1:])))
        # 4b) mejores conexiones inter‑isla (criterio dr>0, dθ<0, distancia mínima)
        for i, ske in enumerate(skeletons):
            if not ske:
                continue
            a = ske[-1]
            best, bd = None, math.inf
            for j, ske2 in enumerate(skeletons):
                if i == j or not ske2:
                    continue
                b = ske2[0]
                dr, dtheta = b[1] - a[1], b[0] - a[0]
                if dr > 0 and dtheta < 0:
                    d = math.hypot(dr, dtheta)
                    if d < bd:
                        bd, best = d, (a, b)
            if best:
                segs.append(best)
        return segs

    # ------------------------------------------------------------------
    # Paso 5 – Construir rutas (contours) y filtrar por Δr
    # ------------------------------------------------------------------
    def _build_contours(self, segs, hist_s, te, re):
        contours: List[List[Tuple[float, float]]] = []
        connections: List[Connection] = []
        for a, b in segs:
            path = self._density_path(a, b, hist_s, te, re)
            route = [a] + path + [b]
            contours.append(route)
            # registrar Δr entre a y b (para estadística global)
            connections.append(Connection(a, b, abs(b[1] - a[1]),
                                           math.hypot(b[0] - a[0], b[1] - a[1])))
        return contours, connections

    @staticmethod
    def _split_by_threshold(contours, thr):
        """Parte las rutas cuando Δr > thr."""
        out = []
        for cnt in contours:
            tmp = [cnt[0]]
            for a, b in zip(cnt, cnt[1:]):
                if abs(b[1] - a[1]) <= thr:
                    tmp.append(b)
                else:
                    if len(tmp) > 1:
                        out.append(tmp)
                    tmp = [b]
            if len(tmp) > 1:
                out.append(tmp)
        return out

    # ------------------------------------------------------------------
    #  Método público – run()  (devuelve skeletons_union)
    # ------------------------------------------------------------------
    def run(self, datos: Dict[str, Any]):
        """
        Ejecuta todo el segmentador y retorna, entre otros,
        ``skeletons_union``: lista de brazos completos
        (skeleton + centroides extra + skeleton enlazado).
        """
        # 0) Puntos de entrada ------------------------------------------------
        bg  = datos["background"][["theta", "r"]].values
        grp = np.vstack([g["points"][["theta", "r"]].values
                         for g in datos["grupos_finales"]])
        pts = np.vstack([bg, grp])

        # 1) Histograma suavizado --------------------------------------------
        hist_s, te, re = self._histogram(pts)

        # 2–3) BFS  → clusters  → skeletons ----------------------------------
        uf, labels          = self._weighted_bfs(hist_s, te, re)
        clusters, skeletons = self._clusters_and_skeletons(pts, uf, labels, te, re)

        # 4) Conexiones + centroides extra -----------------------------------
        segs = self._candidate_segments(skeletons)
        raw_contours, conns = self._build_contours(segs, hist_s, te, re)

        # 5) Filtrado Δr ------------------------------------------------------
        mean_dr = float(np.mean([c.delta_r for c in conns])) if conns else 0.0
        thr     = mean_dr * self.dr_multiplier
        contours = self._split_by_threshold(raw_contours, thr)

        # 6) Objetos ----------------------------------------------------------
        objects = [IslandObject("isla-conexion-isla", cnt) for cnt in contours]
        for c in conns:
            if c.delta_r > thr:
                objects.append(IslandObject("isla-conexiones-adicionales",
                                            [c.a, c.b]))

        # 7) Construir skeletons_union ---------------------------------------
        n_sk   = len(skeletons)
        parent = list(range(n_sk))

        def find_sk(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union_sk(a, b):
            ra, rb = find_sk(a), find_sk(b)
            if ra != rb:
                parent[rb] = ra

        # Mapear extremos -> índice de skeleton
        first_idx = {tuple(sk[0]): i for i, sk in enumerate(skeletons) if sk}
        last_idx  = {tuple(sk[-1]): i for i, sk in enumerate(skeletons) if sk}

        # Guardar los puntos de cada contorno que une skeletons diferentes
        edge_pts: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for cnt in contours:          # cada contorno ya incluye centroides extra
            a, b = cnt[0], cnt[-1]
            i, j = last_idx.get(tuple(a)), first_idx.get(tuple(b))
            if i is None or j is None or i == j:
                continue
            union_sk(i, j)
            edge_pts.setdefault((i, j), []).extend(cnt)

        # Componentes conexos del union-find
        comps: dict[int, list[int]] = {}
        for k in range(n_sk):
            comps.setdefault(find_sk(k), []).append(k)

        # Construir lista final de puntos por componente
        skeletons_union: list[list[tuple[float, float]]] = []
        for idxs in comps.values():
            if not idxs:
                continue
            pts_comp: list[tuple[float, float]] = []
            # a) puntos de todos los skeletons del grupo
            for k in idxs:
                pts_comp.extend(skeletons[k])
            # b) puntos intermedios de todos los contornos internos al grupo
            for i in idxs:
                for j in idxs:
                    if i < j and (i, j) in edge_pts:
                        pts_comp.extend(edge_pts[(i, j)])
            skeletons_union.append(pts_comp)
        
        # --- ordenar skeletons_union por tamaño descendente y quedarme con los 3 primeros ---
        sorted_union = sorted(skeletons_union, key=lambda sk: len(sk), reverse=True)[:4]

        # --- guardar cada uno en un CSV distinto ---
        for i, sk in enumerate(sorted_union, start=1):
            df_sk = pd.DataFrame(sk, columns=["theta", "r"])
            df_sk.to_csv(f"arms_union_top{i}.csv", index=False)

        print(f"[run] skeletons_union groups: {len(skeletons_union)}")

        # 8) Retorno ----------------------------------------------------------
        return {
            "hist_s": hist_s,
            "te": te,
            "re": re,
            "clusters": clusters,
            "skeletons": skeletons,
            "contours": contours,
            "connections": conns,
            "objects": objects,
            "skeletons_union": sorted_union,
            "mean_dr": mean_dr,
            "threshold": thr,
        }


    # ------------------------------------------------------------------
    # 6 – Ajuste de línea para cada objeto (con χ² reducido y PA)
    # ------------------------------------------------------------------    
    def fit_skeletons(self, skeletons: List[List[Tuple[float, float]]]) -> List[Dict[str,float]]:
        """
        Ajusta una recta a cada lista de centroides (skeleton).
        Devuelve lista de dicts con slope, intercept, chi2_reduced, pa,
        theta_min y theta_max para cada skeleton.
        """
        fits = []
        for ske in skeletons:
            arr = np.array(ske)  # shape (N,2): theta, r
            if arr.shape[0] < 2:
                continue
            df = pd.DataFrame(arr, columns=["theta","r"])
            m, b, pa, θ_min, θ_max, _r_min, _r_max = fit_line_to_cluster(df)
            resid = df["r"] - (m * df["theta"] + b)
            chi2  = float((resid**2).sum() / max(len(df)-2,1))
            fits.append({
                "slope": m,
                "intercept": b,
                "chi2": chi2,
                "pa": pa,
                "theta_min": θ_min,
                "theta_max": θ_max,
                "r_min": _r_min,
                "r_max": _r_max,
            })
        return fits




    # ------------------------------------------------------------------
    # 7 – Visualización del ajuste lineal sobre skeletons
    # ------------------------------------------------------------------
    def quick_plot_ajuste(
        self,
        result: Dict[str, Any],
        datos: Dict[str, Any],
        id_halo: str = "",
        figsize: Tuple[int, int] = (10, 6),
        save_path: str = "",
    ):
        """
        Dibuja, para cada camino de centroides (skeleton):
          • Scatter θ vs. r de los centroides
          • Línea de ajuste con leyenda: ecuación, χ² reducido y PA
        """
        # 1) Extraer skeletons y calcular ajustes
        skeletons = result.get("skeletons_union", [])
        fits      = self.fit_skeletons(skeletons)

        # Crear figura
        fig, ax = plt.subplots(figsize=figsize)

        # puntos 
        bg  = datos["background"][["theta", "r"]].values
        grp = np.vstack([g["points"][["theta", "r"]].values
                         for g in datos["grupos_finales"]])
        pts = np.vstack([bg, grp])
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=1.8, color="gray", edgecolor="#eaeaea", alpha=0.12,
        )

        for idx, (ske, fit) in enumerate(zip(skeletons, fits), start=1):
            arr = np.array(ske)            # array (N,2) de (theta, r)
            θ   = arr[:, 0]
            r   = arr[:, 1]
            label = f"Path {idx}"
            ax.scatter(θ, r, s=8, alpha=0.6)

            # 3) Línea de ajuste
            tmin, tmax = fit["theta_min"], fit["theta_max"]
            tvals = np.linspace(tmin, tmax, 100)
            rhat  = fit["slope"] * tvals + fit["intercept"]
            legend_label = (
                rf"$r = {fit['slope']:.3f}\,\theta + {fit['intercept']:.3f}$" "\n"
                rf"$\chi^2_{{\rm red}} = {fit['chi2']:.2f},\;\mathrm{{PA}} = {fit['pa']:.2f}^\circ$"
            )
            ax.plot(tvals, rhat, '-', linewidth=2, label=legend_label)

        # 4) Estética
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel(r"$r$ (kpc)")
        ax.set_title(f"Halo {id_halo} – Centroid Paths Linear Fit")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.001, 1.0),   # (x, y) en ejes (x > 1 sale fuera)
            borderaxespad=0.0,
            fontsize="small",
            framealpha=0.68,
            )
        #ax.grid(True, alpha=0.3)
        _save_or_show(fig, save_path)


    # ------------------------------------------------------------------
    #  Utilidad rápida para visualizar (save option added)
    # ------------------------------------------------------------------
    def quick_plot_pasada(self, result, *, save_path: str | None = None, figsize=(14, 6)):
        """
        Muestra el histograma suavizado con los segmentos de brazo, esqueletos y contornos.
        Añade etiquetas de ejes, título, leyenda y cuadrícula para una visualización clara.
        """
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=figsize)

        # Mostrar histograma suavizado
        im = ax.imshow(
            result["hist_s"].T,
            origin="lower",
            extent=[
                result["te"][0],
                result["te"][-1],
                result["re"][0],
                result["re"][-1],
            ],
            aspect="auto",
            cmap="inferno",
        )
        # Barra de color
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Smoothed density")

        # Dibujar segmentos de brazo
        for idx, cl in enumerate(result["clusters"], start=1):
            ax.scatter(
                cl[:, 0],
                cl[:, 1],
                s=8,
                alpha=0.6,
                label=f"Arm Segment {idx}",
            )


        # Dibujar esqueletos sin añadir a la leyenda
        for ske in result.get("skeletons_union", []):
            arr = np.array(ske)
            if arr.size:
                ax.scatter(
                    arr[:, 0],
                    arr[:, 1],
                    c="w",
                    s=40,
                    edgecolors="k",
                )

        # Dibujar contornos sin añadir a la leyenda
        for cnt in result.get("contours", []):
            xs, ys = zip(*cnt)
            ax.plot(
                xs, ys,
                "-o", color="cyan", markersize=5,
                label="_nolegend_"
            )

        # Etiquetas de ejes y título
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel(r"$r$ (kpc)")
        #ax.set_title(f"Halo {result.get('id_halo', '')} – Histogram & Arm Segments")

        # Leyenda y cuadrícula
        ax.legend(loc="best", fontsize="small", framealpha=0.8)
        #ax.grid(True, alpha=0.3)
        _save_or_show(fig, save_path)

    # ------------------------------------------------------------------
    #  Utilidad rápida para visualizar (save option added, with object fits)
    # ------------------------------------------------------------------
    def quick_plot(self, result, *, save_path: str | None = None, figsize=(14, 6)):
        """
        Displays the smoothed histogram with:
        • Arm segments (clusters)
        • Centroid paths (skeletons)
        • Contours
        • Linear fits over each skeleton path (PA and χ²)
        Adds axis labels, legend and grid.
        """
        # 1) Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # 2) Plot smoothed histogram
        im = ax.imshow(
            result["hist_s"].T,
            origin="lower",
            extent=[
                result["te"][0], result["te"][-1],
                result["re"][0], result["re"][-1],
            ],
            aspect="auto",
            cmap="inferno",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Smoothed density")

        palette = cm.get_cmap("tab10")
        colors = palette.colors 

        # 3) Draw arm segments (clusters)
        for idx, cl in enumerate(result.get("clusters", []), start=1):
            ax.scatter(
                cl[:, 0], cl[:, 1],
                s=8, alpha=0.5,
                color=colors[(idx-1)%10],
                label="_nolegend_"
            )

        # 4) Draw centroid paths (skeletons)
        for idx, ske in enumerate(result.get("skeletons", []), start=1):
            arr = np.array(ske)
            if arr.size:
                ax.scatter(
                    arr[:, 0], arr[:, 1],
                    marker="x", s=20, alpha=0.8,
                    label="_nolegend_"
                )

        # 5) Draw contours (no legend)
        for cnt in result.get("contours", []):
            xs, ys = zip(*cnt)
            ax.plot(
                xs, ys,
                "-o", color="cyan", markersize=4,
                alpha = 0.69, label="_nolegend_"
            )

        # 6) Linear fits on skeletons
        fits = self.fit_skeletons(result.get("skeletons_union", []))
        for idx, fit in enumerate(fits, start=1):
            tmin, tmax = fit["theta_min"], fit["theta_max"]
            tvals = np.linspace(tmin, tmax, 200)
            r_line = fit["slope"] * tvals + fit["intercept"]
            ax.plot(
                tvals, r_line,
                "-", linewidth=3, color="white",  # silver halo
                alpha=0.6,
                zorder=2,
                label="_nolegend_"
            )
            ax.plot(
                tvals, r_line,
                "-", color=colors[(idx-1)%10],
                linewidth=1.5, alpha=0.98,
                zorder=5,
                label=f"Arm Segment {idx} fit: PA={fit['pa']:.1f}°, χ²={fit['chi2']:.1f}"
            )

            line = ax.plot(
                tvals, r_line,
                "-", color=colors[(idx-1)%10],
                linewidth=1.8, alpha=0.8,
                zorder=4,
            )[0]
            line.set_path_effects([
                PathEffects.withStroke(linewidth=4, foreground="white", alpha=0.87)
            ])

        # 7) Aesthetics
        ax.set_xlabel(r"$\theta$ (deg)")
        ax.set_ylabel(r"$r$ (kpc)")
        #ax.set_title(f"Halo {result.get('id_halo','')} – Histogram & Skeleton Fits")
        ax.legend(loc="best", fontsize="small", framealpha=0.8)
        #ax.grid(True, alpha=0.3)

        # 8) Save or show
        _save_or_show(fig, save_path)


# -----------------------------------------------------------------------------
# 9 – PIPELINE MAIN
# -----------------------------------------------------------------------------

def pipeline_for_id(
    id_subhalo: str | int,
    *,
    output_dir: str = "figures",
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Run the full pipeline for a single *id_subhalo*.

    Returns the final *datos* dictionary for downstream use.
    """
    _ensure_dir(output_dir)

    bfs, df_f = generate_bfs_seeds(
        id_halo=id_subhalo,
        theta_min=30,
        theta_max=540,
        quartile_threshold=0.10,
        theta_diff=3.0,
        r_diff=0.5,
        gap_threshold_theta=1.80,
        gap_threshold_r=1.0,
        file_prefix="data_rho",
    )

    merged = adjust_and_merge_seeds(bfs, slope_variation_threshold=0.30, bounding_extrap=0.35)
    validated = validate_dispersion_and_reprocess(merged)

    # Visualisations + saving
    cart_path = os.path.join(output_dir, f"cartesian_groups_{id_subhalo}.png") if save_plots else None
    polar_path = os.path.join(output_dir, f"polar_groups_{id_subhalo}.png") if save_plots else None
    plot_groups_cartesian(validated, df_f, save_path=cart_path)
    plot_groups_polar(validated, df_f, save_path=polar_path)

    # Keep only big groups
    grupos_finales = [g for g in validated if len(g["points"]) >= 60]
    grupos_finales = sorted(
        grupos_finales,
        key=lambda g: len(g["points"]),
        reverse=True
    )
    puntos_grupos = {
        p_id for g in grupos_finales for p_id in g["points"]["id"].tolist()
    }
    background = df_f[~df_f["id"].isin(puntos_grupos)].copy()

    print(
        f"Halo {id_subhalo}: {len(grupos_finales)} final groups (≥60 pts) | "
        f"background pts: {len(background)}",
    )
    return {"grupos_finales": grupos_finales, "background": background}


import matplotlib.patheffects as PathEffects
import matplotlib.patheffects as PathEffects

def plot_islands_and_paths_polar_pasada(
    result: Dict[str, Any],
    datos: Dict[str, Any],
    ajustes: List[Dict[str, float]],
    id_halo: str,
    line_extrap: float = 0.05,
    save_path: str | None = None,
) -> None:
    """
    Gráfico polar que muestra:
      • islas ≥ min_cluster_size
      • caminos de centroides con borde
      • contornos
    """
    # 1) Filtrar islas grandes
    big_islands = [cl for cl in result['clusters'] if len(cl) >= 60]
    print(f"Islas ≥60 puntos (polar): {len(big_islands)}")

    # 2) Preparar figura polar
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="polar")

    # 3) Dibujar puntos de fondo
    bg = datos["background"][["theta", "r"]].values
    ax.scatter(
        np.radians(bg[:, 0]), bg[:, 1],
        s=3, color="gray", edgecolor="black", alpha=0.2,
        label="Data points"
    )
    
    

    colors = ["red","blue","green","purple","orange","brown",
              "magenta","cyan","gold","gray","lime"]

    # 4) Para cada isla y su ajuste correspondiente
    for idx, (cl, fit) in enumerate(zip(big_islands, ajustes)):
        c = colors[idx % len(colors)]
        θ = np.radians(cl[:, 0])
        r = cl[:, 1]
        ax.scatter(θ, r, s=10, color=c, alpha=0.4)

        # 4a) Caminos de centroides con borde negro
        for ske in result.get("skeletons", []):
            arr = np.array(ske)
            if arr.size:
                θ_s = np.radians(arr[:, 0])
                r_s = arr[:, 1]
                lines = ax.plot(θ_s, r_s, color="yellow", linewidth=2, alpha=0.8)
                for ln in lines:
                    ln.set_path_effects([
                        PathEffects.withStroke(linewidth=3, alpha=0.56, foreground="black")
                    ])

        # 4b) Contornos
        for cnt in result.get("contours", []):
            θ_cnt, r_cnt = zip(*cnt)
            ax.plot(
                np.radians(θ_cnt), r_cnt,
                "o-", color="cyan", markersize=2, alpha=0.6
            )

        # 4c) Línea de ajuste extrapolada
        tmin, tmax = fit["theta_min"], fit["theta_max"]
        dt = tmax - tmin
        tvals = np.linspace(tmin - line_extrap*dt, tmax + line_extrap*dt, 200)
        r_line = fit["slope"] * tvals + fit["intercept"]
        ax.plot(
            np.radians(tvals), r_line,
            "--", color=c, linewidth=2, alpha=0.7,
            label = (f"Arm segment {idx+1} fit: PA ={fit['pa']:.1f}°, χ² ={fit['chi2']:.1f}")
        )
        ax.plot(
            np.radians(tvals), r_line,
            "-", color="black", linewidth=2.5, alpha=0.37
        )

    # 5) Ajustes finales del gráfico
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_title(f"Halo {id_halo} – Polar con Ajuste", y=1.05)
    ax.grid(True, alpha=0.1)
    ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.03, 1.0),   # (x, y) en ejes (x > 1 sale fuera)
    borderaxespad=0.0,
    fontsize="small",
    framealpha=0.8,
    )
    _save_or_show(fig, save_path)


def plot_islands_and_paths_polar(
    result: Dict[str, Any],
    datos: Dict[str, Any],
    ajustes: List[Dict[str, float]],
    id_halo: str,
    line_extrap: float = 0.05,
    save_path: str | None = None,
) -> None:
    """
    Polar plot displaying:
      • Islands with ≥ min_cluster_size
      • Centroid paths in black
      • Contours
      • Linear fits in colored dashed lines with black borders
    """
    # 1) Filter large islands
    big_islands = [cl for cl in result['clusters'] if len(cl) >= 60]
    big_islands = sorted(big_islands, key=lambda cl: len(cl), reverse=True)

    print(f"Islands ≥60 points (polar): {len(big_islands)}")

    # 2) Prepare polar figure
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("white")  # Light gray for readability

    # 3) Background points
    bg = datos["background"][["theta", "r"]].values
    ax.scatter(
        np.radians(bg[:, 0]), bg[:, 1],
        s=3, color="gray", edgecolor="black", alpha=0.2,
        label="Background points"
    )

    colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
              "#A65628", "#F781BF", "#999999", "#66C2A5", "#FFD92F"]

    # 4) Plot each island and its corresponding fit
    for idx, (cl, fit) in enumerate(zip(big_islands, ajustes)):
        c = colors[idx % len(colors)]
        θ = np.radians(cl[:, 0])
        r = cl[:, 1]
        ax.scatter(θ, r, s=10, color=c, alpha=0.4)

        # 4a) Centroid paths (black lines)
        for ske in result.get("skeletons", []):
            arr = np.array(ske)
            if arr.size:
                θ_s = np.radians(arr[:, 0])
                r_s = arr[:, 1]
                ax.plot(θ_s, r_s, color="gray", linewidth=1.5, alpha=0.7)

        # 4b) Contours
        for cnt in result.get("contours", []):
            θ_cnt, r_cnt = zip(*cnt)
            ax.plot(
                np.radians(θ_cnt), r_cnt,
                "o-", color="cyan", markersize=2, alpha=0.36
            )

        # 4c) Linear fit with border using path effects
        tmin, tmax = fit["theta_min"], fit["theta_max"]
        dt = tmax - tmin
        tvals = np.linspace(tmin - line_extrap * dt, tmax + line_extrap * dt, 200)
        r_line = fit["slope"] * tvals + fit["intercept"]
        θ_fit = np.radians(tvals)
        ax.plot(
            np.radians(tvals), r_line,
            "-", color="black", linewidth=5.0, alpha=0.77
        )
        line = ax.plot(
            θ_fit, r_line,
            "--", color=c, linewidth=1.8, alpha=0.9,
            label=f"Arm segment {idx+1} fit: PA={fit['pa']:.1f}°, χ²={fit['chi2']:.1f}",
            zorder=3
        )[0]
        line.set_path_effects([
            PathEffects.withStroke(linewidth=2.2, foreground=c, alpha=0.7)
        ])

    # 5) Final plot adjustments
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_title(f"Halo {id_halo} – Polar Fit View", y=1.05)
    ax.grid(True, alpha=0.15)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0.0,
        fontsize="small",
        framealpha=0.8,
    )

    # Show or save
    _save_or_show(fig, save_path)


from typing import Any, Dict, List, Optional
from typing import Any, Dict, List, Optional

def plot_islands_and_paths_polar_intercep(
    result: Dict[str, Any],
    datos: Dict[str, Any],
    ajustes: List[Dict[str, float]],
    id_halo: str,
    min_cluster_size: int = 60,
    line_extrap: float = 0.05,
    dup_threshold: float = 0.90,
    r_proximity: float = 0.05,
    save_path: Optional[str] = None,
) -> None:
    """
    Polar visualization of halo data:
      - Clusters ≥ min_cluster_size, remove overlapping clusters
      - Longest linear fits, remove intersecting/overlapping and contained+close fits
      - Plot skeletons, contours, clusters, fits (legends only for fits)
    """
    # Background handling
    bg_raw = datos.get("background", None)
    theta_bg = np.array([])
    r_bg = np.array([])
    if bg_raw is not None:
        try:
            theta_bg = np.radians(bg_raw["theta"].values)
            r_bg = bg_raw["r"].values
        except Exception:
            arr = np.asarray(bg_raw)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                theta_bg = np.radians(arr[:, 0])
                r_bg = arr[:, 1]

    # ─────────── Clusters (Condition 1 + dedup) ───────────
    clusters = result.get("clusters", [])
    idx_cl = [i for i, cl in enumerate(clusters) if len(cl) >= min_cluster_size]
    idx_cl.sort(key=lambda i: len(clusters[i]), reverse=True)
    clusters_f = [clusters[i] for i in idx_cl]
    cluster_r_sets = [set(np.round(cl[:,1],3)) for cl in clusters_f]
    rm_cl = set()
    for i in range(len(clusters_f)):
        for j in range(i+1, len(clusters_f)):
            if i in rm_cl or j in rm_cl: continue
            Ri, Rj = cluster_r_sets[i], cluster_r_sets[j]
            if len(Ri & Rj) / float(min(len(Ri), len(Rj))) >= dup_threshold:
                rm_cl.add(i if len(clusters_f[i]) < len(clusters_f[j]) else j)
    clusters_clean = [cl for idx,cl in enumerate(clusters_f) if idx not in rm_cl]
    # extra big island
    delta_r_big = 1.0
    if clusters_clean:
        max_pts = len(clusters_clean[0])
        ranges_cl = [(cl[:,1].min(), cl[:,1].max()) for cl in clusters_clean]
        for cl in clusters:
            if len(cl) <= max_pts: continue
            rmin, rmax = cl[:,1].min(), cl[:,1].max()
            if all(rmax + delta_r_big < lo or rmin - delta_r_big > hi for lo,hi in ranges_cl):
                clusters_clean.append(cl)
                break

    # ─────────── Fits (Conditions 2–6) ───────────
    fits_raw = [ajustes[i] for i in idx_cl if i < len(ajustes)]
    fits_info = []
    for fit in fits_raw:
        m, c = fit.get("slope",0), fit.get("intercept",0)
        tmin, tmax = fit.get("theta_min",0), fit.get("theta_max",0)
        r1, r2 = m*tmin + c, m*tmax + c
        length = np.hypot(r2*np.cos(np.radians(tmax)) - r1*np.cos(np.radians(tmin)),
                          r2*np.sin(np.radians(tmax)) - r1*np.sin(np.radians(tmin)))
        fits_info.append({**fit, "length":length, "r_min":min(r1,r2), "r_max":max(r1,r2)})
    # remove intersections & overlaps
    rm_fit = set()
    n_fit = len(fits_info)
    for i in range(n_fit):
        for j in range(i+1, n_fit):
            if i in rm_fit or j in rm_fit: continue
            fi, fj = fits_info[i], fits_info[j]
            mi, ci = fi["slope"], fi["intercept"]
            mj, cj = fj["slope"], fj["intercept"]
            # angular intersection
            if mi != mj:
                ti = (cj - ci)/(mi - mj)
                if fi["theta_min"]<=ti<=fi["theta_max"] and fj["theta_min"]<=ti<=fj["theta_max"]:
                    rm_fit.add(i if fi["length"]<fj["length"] else j); continue
            # r-overlap
            Ri = set(np.round(np.linspace(fi["r_min"],fi["r_max"],100),3))
            Rj = set(np.round(np.linspace(fj["r_min"],fj["r_max"],100),3))
            if len(Ri & Rj)/float(min(len(Ri),len(Rj)))>=dup_threshold:
                rm_fit.add(i if fi["length"]<fj["length"] else j)
    fits_clean = [f for idx,f in enumerate(fits_info) if idx not in rm_fit]
    # extra long isolated
    if fits_clean:
        max_len = max(f["length"] for f in fits_clean)
        ranges = [(f["r_min"],f["r_max"]) for f in fits_clean]
        for f in fits_info:
            if f in fits_clean or f["length"]<=max_len: continue
            if all(f["r_max"]+delta_r_big<lo or f["r_min"]-delta_r_big>hi for lo,hi in ranges):
                fits_clean.append(f); break
    # containment + proximity removal (Condition 6)
    rm_close = set()
    for i in range(len(fits_clean)):
        for j in range(i+1, len(fits_clean)):
            if i in rm_close or j in rm_close: continue
            fi, fj = fits_clean[i], fits_clean[j]
            # full containment
            cont_i_in_j = (fi["theta_min"]>=fj["theta_min"] and fi["theta_max"]<=fj["theta_max"]
                           and fi["r_min"]>=fj["r_min"] and fi["r_max"]<=fj["r_max"])
            cont_j_in_i = (fj["theta_min"]>=fi["theta_min"] and fj["theta_max"]<=fi["theta_max"]
                           and fj["r_min"]>=fi["r_min"] and fj["r_max"]<=fi["r_max"])
            if cont_i_in_j or cont_j_in_i:
                # minimal r-distance
                dr = 0
                if fi["r_max"] < fj["r_min"]: dr = fj["r_min"] - fi["r_max"]
                elif fj["r_max"] < fi["r_min"]: dr = fi["r_min"] - fj["r_max"]
                if dr <= r_proximity:
                    # remove shorter
                    rm_close.add(i if fi["length"]<fj["length"] else j)
    fits_clean = [f for idx,f in enumerate(fits_clean) if idx not in rm_close]
    # sort and assign
    fits_clean.sort(key=lambda f: f["length"], reverse=True)
    for idx, f in enumerate(fits_clean): f["idx"]=idx

    # ─────────── Plotting ───────────
    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(9,8))
    ax.set_facecolor("white")
    # background (no legend)
    if theta_bg.size:
        ax.scatter(theta_bg, r_bg, s=3, color="gray", alpha=0.2)
    colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#A65628","#F781BF","#999999","#66C2A5","#FFD92F"]
    # clusters (no legend)
    for idx, cl in enumerate(clusters_clean):
        ax.scatter(np.radians(cl[:,0]), cl[:,1], s=8, color=colors[idx%len(colors)], alpha=0.6)
    # skeletons & contours
    # for ske in result.get("skeletons",[]):
    #     arr = np.array(ske)
    #     if arr.size: ax.plot(np.radians(arr[:,0]), arr[:,1], color="gray", lw=1.5, alpha=0.7)
    # for cnt in result.get("contours",[]):
    #     th, rr = zip(*cnt)
    #     ax.plot(np.radians(th), rr, 'o-', color="cyan", ms=2, alpha=0.36)
    # fits (with legend)
    for f in fits_clean:
        idx, m, c = f["idx"], f["slope"], f["intercept"]
        tmin, tmax = f["theta_min"], f["theta_max"]
        dv = tmax-tmin
        tvals = np.linspace(tmin-dv*line_extrap, tmax+dv*line_extrap,200)
        rvals = m*tvals + c
        th = np.radians(tvals)
        ax.plot(th, rvals, '-', color='black', lw=5, alpha=0.77)
        ln = ax.plot(th, rvals, '--', color=colors[idx%len(colors)], lw=1.8, alpha=0.9,
                     label=f"Fit {idx+1}: PA={f.get('pa',0):.1f}°, χ²={f.get('chi2',0):.1f}")[0]
        ln.set_path_effects([PathEffects.withStroke(linewidth=2.2, foreground=colors[idx%len(colors)], alpha=0.7)])
    # aesthetics
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title(f"Halo {id_halo} – Polar Fit View", y=1.05)
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03,1.0), fontsize="small", framealpha=0.8)
    _save_or_show(fig, save_path)



def plot_islands_and_paths_polar_intercep_t(
    result: Dict[str, Any],
    datos: Dict[str, Any],
    ajustes: List[Dict[str, float]],
    id_halo: str,
    min_cluster_size: int = 60,
    line_extrap: float = 0.05,
    dup_threshold: float = 0.90,
    r_proximity: float = 0.05,
    save_path: Optional[str] = None,
) -> None:
    """
    Polar visualization of halo data:
      - Clusters ≥ min_cluster_size, remove overlapping clusters
      - Longest linear fits, remove intersecting/overlapping and contained+close fits
      - Plot skeletons, contours, clusters, fits (legends only for fits)
      - Only display the first two fits
    """
    # Background handling
    bg_raw = datos.get("background", None)
    theta_bg = np.array([])
    r_bg = np.array([])
    if bg_raw is not None:
        try:
            theta_bg = np.radians(bg_raw["theta"].values)
            r_bg = bg_raw["r"].values
        except Exception:
            arr = np.asarray(bg_raw)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                theta_bg = np.radians(arr[:, 0])
                r_bg = arr[:, 1]

    # ─────────── Clusters (Condition 1 + dedup) ───────────
    clusters = result.get("clusters", [])
    idx_cl = [i for i, cl in enumerate(clusters) if len(cl) >= min_cluster_size]
    idx_cl.sort(key=lambda i: len(clusters[i]), reverse=True)
    clusters_f = [clusters[i] for i in idx_cl]
    cluster_r_sets = [set(np.round(cl[:,1],3)) for cl in clusters_f]
    rm_cl = set()
    for i in range(len(clusters_f)):
        for j in range(i+1, len(clusters_f)):
            if i in rm_cl or j in rm_cl: continue
            Ri, Rj = cluster_r_sets[i], cluster_r_sets[j]
            if len(Ri & Rj) / min(len(Ri), len(Rj)) >= dup_threshold:
                rm_cl.add(i if len(clusters_f[i]) < len(clusters_f[j]) else j)
    clusters_clean = [cl for idx, cl in enumerate(clusters_f) if idx not in rm_cl]
    # Extra big island (Condition 5)
    delta_r_big = 1.0
    if clusters_clean:
        max_pts = len(clusters_clean[0])
        ranges_cl = [(cl[:,1].min(), cl[:,1].max()) for cl in clusters_clean]
        for cl in clusters:
            if len(cl) <= max_pts: continue
            rmin, rmax = cl[:,1].min(), cl[:,1].max()
            if all(rmax + delta_r_big < lo or rmin - delta_r_big > hi for lo,hi in ranges_cl):
                clusters_clean.append(cl)
                break

    # ─────────── Fits (Conditions 2–6) ───────────
    fits_raw = [ajustes[i] for i in idx_cl if i < len(ajustes)]
    fits_info: List[Dict[str, Any]] = []
    for fit in fits_raw:
        m, c = fit.get("slope",0), fit.get("intercept",0)
        tmin, tmax = fit.get("theta_min",0), fit.get("theta_max",0)
        # sample along fit for accurate polyline length
        tvals_len = np.linspace(tmin, tmax, 200)
        r_line_len = m * tvals_len + c
        x_vals = r_line_len * np.cos(np.radians(tvals_len))
        y_vals = r_line_len * np.sin(np.radians(tvals_len))
        length = np.sum(np.hypot(np.diff(x_vals), np.diff(y_vals)))
        # compute end radii
        r1, r2 = m * tmin + c, m * tmax + c
        fits_info.append({**fit, "length": length, "r_min": min(r1, r2), "r_max": max(r1, r2)})
    # remove intersections & overlaps
    rm_fit = set()
    n_fit = len(fits_info)
    for i in range(n_fit):
        for j in range(i+1, n_fit):
            if i in rm_fit or j in rm_fit: continue
            fi, fj = fits_info[i], fits_info[j]
            mi, ci = fi["slope"], fi["intercept"]
            mj, cj = fj["slope"], fj["intercept"]
            if mi != mj:
                ti = (cj - ci) / (mi - mj)
                if fi["theta_min"] <= ti <= fi["theta_max"] and fj["theta_min"] <= ti <= fj["theta_max"]:
                    rm_fit.add(i if fi["length"] < fj["length"] else j)
                    continue
            Ri = set(np.round(np.linspace(fi["r_min"], fi["r_max"], 100),3))
            Rj = set(np.round(np.linspace(fj["r_min"], fj["r_max"], 100),3))
            if len(Ri & Rj) / min(len(Ri), len(Rj)) >= dup_threshold:
                rm_fit.add(i if fi["length"] < fj["length"] else j)
    fits_clean = [f for idx, f in enumerate(fits_info) if idx not in rm_fit]
    # extra long isolated
    if fits_clean:
        max_len = max(f["length"] for f in fits_clean)
        ranges = [(f["r_min"], f["r_max"]) for f in fits_clean]
        for f in fits_info:
            if f in fits_clean or f["length"] <= max_len: continue
            if all(f["r_max"] + delta_r_big < lo or f["r_min"] - delta_r_big > hi for lo,hi in ranges):
                fits_clean.append(f)
                break
    # containment + proximity
    rm_close = set()
    for i in range(len(fits_clean)):
        for j in range(i+1, len(fits_clean)):
            if i in rm_close or j in rm_close: continue
            fi, fj = fits_clean[i], fits_clean[j]
            theta_overlap = max(0, min(fi["theta_max"], fj["theta_max"]) - max(fi["theta_min"], fj["theta_min"]))
            theta_base = min(fi["theta_max"] - fi["theta_min"], fj["theta_max"] - fj["theta_min"]) or 1
            frac_theta = theta_overlap / theta_base
            r_overlap = max(0, min(fi["r_max"], fj["r_max"]) - max(fi["r_min"], fj["r_min"]))
            r_base = min(fi["r_max"] - fi["r_min"], fj["r_max"] - fj["r_min"]) or 1
            frac_r = r_overlap / r_base
            if frac_theta >= dup_threshold or frac_r >= dup_threshold:
                rm_close.add(i if fi["length"] < fj["length"] else j)
    fits_clean = [f for idx, f in enumerate(fits_clean) if idx not in rm_close]
    # sort and assign idx
    fits_clean.sort(key=lambda f: f["length"], reverse=True)
    for idx, f in enumerate(fits_clean): f["idx"] = idx

    # Only keep first two fits
    fits_display = fits_clean[:2]

    # ─────────── Plotting ───────────
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9,8))
    ax.set_facecolor("white")
    if theta_bg.size:
        ax.scatter(theta_bg, r_bg, s=3, color="gray", alpha=0.2)
    colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#A65628","#F781BF","#999999","#66C2A5","#FFD92F"]
    # clusters (no legend)
    for idx, cl in enumerate(clusters_clean):
        ax.scatter(np.radians(cl[:,0]), cl[:,1], s=8, color=colors[idx % len(colors)], alpha=0.6)
    # fits (with legend)
    for f in fits_display:
        idx, m, c = f["idx"], f["slope"], f["intercept"]
        tmin, tmax = f["theta_min"], f["theta_max"]
        dv = tmax - tmin
        tvals = np.linspace(tmin - dv * line_extrap, tmax + dv * line_extrap, 200)
        rvals = m * tvals + c
        th = np.radians(tvals)
        ax.plot(th, rvals, '-', color='black', lw=5, alpha=0.77)
        ln = ax.plot(th, rvals, '--', color=colors[idx % len(colors)], lw=1.8, alpha=0.9,
                     label=f"Fit {idx+1}: PA={f.get('pa', 0):.1f}°, χ²={f.get('chi2', 0):.1f}")[0]
        ln.set_path_effects([PathEffects.withStroke(linewidth=2.2, foreground=colors[idx % len(colors)], alpha=0.7)])
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title(f"Halo {id_halo} – Polar Fit View", y=1.05)
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize="small", framealpha=0.8)
    _save_or_show(fig, save_path)






# -----------------------------------------------------------------------------
# 10 – CLI ENTRY POINT
# -----------------------------------------------------------------------------


def _cli():
    p = argparse.ArgumentParser(
        description="Run arm-tracing pipeline on one or more subhalos."
    )
    p.add_argument("ids", nargs="+", help="Subhalo IDs to process")
    p.add_argument("--out", default="figures", help="Directory to save figures [figures]")
    p.add_argument("--no-show", action="store_true", help="Save figs but do not display")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for _id in args.ids:
        # 1) Ejecuto pipeline principal
        datos = pipeline_for_id(_id, output_dir=args.out, save_plots=True)

        # — Guardar datos['background'] —
        bg_path = os.path.join(args.out, f"background_{_id}.csv")
        datos["background"].to_csv(bg_path, index=False)
        print(f"background → {bg_path}")

        # — Guardar grupos_finales —
        # Concateno todos los grupos en un solo DF, añadiendo columna 'group_id'
        grupos = datos["grupos_finales"]
        grupos_dfs = []
        for gi, grupo in enumerate(grupos):
            dfp = grupo["points"].copy()
            dfp["group_id"] = gi
            grupos_dfs.append(dfp)
        all_groups = pd.concat(grupos_dfs, ignore_index=True)
        groups_path = os.path.join(args.out, f"groups_{_id}.csv")
        all_groups.to_csv(groups_path, index=False)
        print(f"grupos_finales → {groups_path}")

        # 2) Segmentación e ajustes
        print("voy a entrar al HistogramSegmenter")
        segmenter = HistogramSegmenter(bins_theta=120, bins_r=80, dr_multiplier=2.5)
        result = segmenter.run(datos)

        # — Guardar clusters —
        # Cada 'cluster' es un array Nx2 de (theta, r)
        clusters = result["clusters"]
        clusters_dfs = []
        for ci, cl in enumerate(clusters):
            dfc = pd.DataFrame(cl, columns=["theta", "r"])
            dfc["cluster_id"] = ci
            clusters_dfs.append(dfc)
        all_clusters = pd.concat(clusters_dfs, ignore_index=True)
        clusters_path = os.path.join(args.out, f"clusters_{_id}.csv")
        all_clusters.to_csv(clusters_path, index=False)
        print(f"  • clusters → {clusters_path}")

        # — Guardar skeletons_union —
        su = result["skeletons_union"]
        su_dfs = []
        for si, sk in enumerate(su):
            dfs = pd.DataFrame(sk, columns=["theta", "r"])
            dfs["skeleton_id"] = si
            su_dfs.append(dfs)
        all_su = pd.concat(su_dfs, ignore_index=True)
        su_path = os.path.join(args.out, f"skeletons_union_{_id}.csv")
        all_su.to_csv(su_path, index=False)
        print(f"  • skeletons_union → {su_path}")

        # 3) Ajustes de fitting
        ajustes = segmenter.fit_skeletons(result["skeletons_union"])
        df_aj = pd.DataFrame(ajustes)
        ajustes_path = os.path.join(args.out, f"ajustes_{_id}.csv")
        df_aj.to_csv(ajustes_path, index=False)
        print(f"  • ajustes → {ajustes_path}")

        # 4) Plots y resto de visualizaciones
        seg_path = os.path.join(args.out, f"hist_segment_{_id}.png")
        segmenter.quick_plot(result, save_path=seg_path)

        fit_path = os.path.join(args.out, f"fit_segment_{_id}.png")
        segmenter.quick_plot_ajuste(result, datos, id_halo=_id, save_path=fit_path)

        polar_path = os.path.join(args.out, f"fit_segment_{_id}_polar.png")
        plot_islands_and_paths_polar(
            result, datos, ajustes, id_halo=_id,
            line_extrap=0.070, save_path=polar_path
        )

        polar_path_single = os.path.join(args.out, f"fit_segment_{_id}_polar_single.png")
        plot_islands_and_paths_polar_intercep_t(
            result, datos, ajustes, id_halo=_id,
            line_extrap=0.050, save_path=polar_path_single
        )

        print("─" * 60)

    if args.no_show:
        plt.close("all")

if __name__ == "__main__":
    _cli()
