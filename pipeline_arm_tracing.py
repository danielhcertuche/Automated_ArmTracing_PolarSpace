#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined pipeline script:
- Generates BFS seeds and filters data for given halo IDs
- Adjusts and merges seeds
- Validates dispersion and reprocesses
- Segments histogram with HistogramSegmenter
- Saves cartesian, polar, and histogram plots to files

Usage:
    python pipeline_full.py --ids 418336 123456 --file-prefix data_rho --output-dir results
"""
import argparse
import os
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import deque
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter, binary_closing, label

# ═══════════════════════════════════════════════════════════════════════
# 0) PITCH-ANGLE
# ═══════════════════════════════════════════════════════════════════════
def calculate_pa(slope, intercept):
    if intercept != 0:
        return np.degrees(np.arctan((slope * (180 / np.pi)) / intercept))
    return np.nan

# ═══════════════════════════════════════════════════════════════════════
# 1) Load and filter data
# ═══════════════════════════════════════════════════════════════════════
def load_and_filter_data(id_halo, theta_min, theta_max, file_prefix='data_rho'):
    df = pd.read_csv(f"{file_prefix}_{id_halo}_filtered.csv")
    df['id'] = df.index
    df['r'] = np.sqrt(df['x']**2 + df['y']**2)
    df['theta'] = np.degrees(np.arctan2(df['y'], df['x']))
    df.loc[df['theta'] < 0, 'theta'] += 360
    df = df[(df['theta'] >= theta_min) & (df['theta'] <= theta_max)].copy()
    df['theta'] = df['theta'].astype(float)
    df['r'] = df['r'].astype(float)
    return df

# ═══════════════════════════════════════════════════════════════════════
# 2) Build graph and BFS
# ═══════════════════════════════════════════════════════════════════════
def build_graph_rectangular(df_points, theta_diff, r_diff):
    th, rr = df_points['theta'].values, df_points['r'].values
    n = len(df_points)
    graph = [[] for _ in range(n)]
    for i in range(n):
        mask = (
            (np.abs(th[i] - th) <= theta_diff) &
            (np.abs(rr[i] - rr) <= r_diff) &
            (np.arange(n) != i)
        )
        graph[i] = list(np.where(mask)[0])
    return graph, n


def bfs_components(graph, df_points):
    visited = [False] * len(graph)
    clusters = []
    for s in range(len(graph)):
        if not visited[s]:
            q = deque([s])
            comp = [s]
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

# ═══════════════════════════════════════════════════════════════════════
# 3) Subdivide by gaps
# ═══════════════════════════════════════════════════════════════════════
def subdivide_by_gap(df_cluster, gap_threshold=2.5, mode='theta'):
    col = 'theta' if mode == 'theta' else 'r'
    df = df_cluster.sort_values(col)
    dif = np.diff(df[col].values)
    if np.all(dif <= gap_threshold):
        return [df]
    subs = []
    start = 0
    for i, d in enumerate(dif):
        if d > gap_threshold:
            subs.append(df.iloc[start:i+1].copy())
            start = i + 1
    subs.append(df.iloc[start:].copy())
    return subs

# ═══════════════════════════════════════════════════════════════════════
# 4) Generate BFS seeds
# ═══════════════════════════════════════════════════════════════════════
def _adaptive_factor(n_pts, ref=2000):
    return np.sqrt(max(n_pts / ref, 1.0))


def generate_bfs_seeds(
    id_halo,
    theta_min=50, theta_max=250,
    quartile_threshold=0.55,
    theta_diff=3.0, r_diff=0.5,
    gap_threshold_theta=2.0, gap_threshold_r=2.0,
    file_prefix='data_rho'
):
    df = load_and_filter_data(id_halo, theta_min, theta_max, file_prefix)
    thr = df['rho_resta_final_exp'].quantile(quartile_threshold)
    df_f = df[df['rho_resta_final_exp'] > thr].copy().reset_index(drop=True)

    f = _adaptive_factor(len(df_f))
    th_diff, r_d = theta_diff / f, r_diff / f
    gap_th, gap_r = gap_threshold_theta / f, gap_threshold_r / f

    graph, _ = build_graph_rectangular(df_f, th_diff, r_d)
    clusters = bfs_components(graph, df_f)

    final_cl = []
    for cl in clusters:
        for st in subdivide_by_gap(cl, gap_th, 'theta'):
            final_cl.extend(subdivide_by_gap(st, gap_r, 'r'))
    return final_cl, df_f

# ═══════════════════════════════════════════════════════════════════════
# 5) Fit and merge seeds
# ═══════════════════════════════════════════════════════════════════════
def fit_line_to_cluster(df_cluster):
    if len(df_cluster) < 2:
        return (None,) * 7
    mdl = LinearRegression().fit(df_cluster[['theta']], df_cluster['r'])
    s, b = mdl.coef_[0], mdl.intercept_
    pa = calculate_pa(s, b)
    return s, b, pa, df_cluster['theta'].min(), df_cluster['theta'].max(), df_cluster['r'].min(), df_cluster['r'].max()


def adjust_and_merge_seeds(bfs_clusters, slope_variation_threshold=0.40, bounding_extrap=0.30):
    groups = []
    for cl in bfs_clusters:
        groups.append(dict(zip(
            ['slope','intercept','pa','theta_min','theta_max','r_min','r_max','points'],
            (*fit_line_to_cluster(cl), cl)
        )))

    def recalc(g):
        g['slope'], g['intercept'], g['pa'], g['theta_min'], g['theta_max'], g['r_min'], g['r_max'] = \
            fit_line_to_cluster(g['points'])

    def boxes_overlap(g1, g2, e):
        def exp(t0,t1,r0,r1):
            dt, dr = (t1-t0)*e, (r1-r0)*e
            return t0-dt, t1+dt, r0-dr, r1+dr
        a = exp(g1['theta_min'],g1['theta_max'],g1['r_min'],g1['r_max'])
        b = exp(g2['theta_min'],g2['theta_max'],g2['r_min'],g2['r_max'])
        return not (a[1]<b[0] or b[1]<a[0]) and not (a[3]<b[2] or b[3]<a[2])

    merged = True
    while merged:
        merged, new = False, []
        i = 0
        while i < len(groups):
            g1, j = groups[i], i+1
            while j < len(groups):
                g2 = groups[j]
                if g1['slope'] is None or g2['slope'] is None:
                    j += 1
                    continue
                if boxes_overlap(g1, g2, bounding_extrap):
                    comb = pd.concat([g1['points'], g2['points']], ignore_index=True)
                    s_comb, b_comb, *_ = fit_line_to_cluster(comb)
                    if s_comb is not None:
                        v1 = abs(s_comb - g1['slope'])/(abs(g1['slope'])+1e-12)
                        v2 = abs(s_comb - g2['slope'])/(abs(g2['slope'])+1e-12)
                        if v1 < slope_variation_threshold and v2 < slope_variation_threshold:
                            g1['points'] = comb
                            recalc(g1)
                            groups.pop(j)
                            merged = True
                            continue
                j += 1
            new.append(g1)
            i += 1
        groups = new
    return [g for g in groups if len(g['points']) >= 2]

# ═══════════════════════════════════════════════════════════════════════
# 6) Validate dispersion and reprocess
# ═══════════════════════════════════════════════════════════════════════
def validate_dispersion_and_reprocess(groups, dispersion_threshold=1.80, reproc_theta_diff=1.25, reproc_r_diff=0.4):
    ok, todo = [], []
    for g in groups:
        if g['slope'] is None or len(g['points']) < 2:
            ok.append(g)
            continue
        res = g['points']['r'] - (g['slope'] * g['points']['theta'] + g['intercept'])
        (todo if res.std() > dispersion_threshold else ok).append(g)

    new = []
    for g in todo:
        subdf = g['points'].reset_index(drop=True)
        graph, _ = build_graph_rectangular(subdf, reproc_theta_diff, reproc_r_diff)
        for c in bfs_components(graph, subdf):
            new.append(dict(zip(
                ['slope','intercept','pa','theta_min','theta_max','r_min','r_max','points'],
                (*fit_line_to_cluster(c), c)
            )))
    return ok + new

# ═══════════════════════════════════════════════════════════════════════
# 7-A) Cartesian plot for large groups
# ═══════════════════════════════════════════════════════════════════════
def plot_groups_cartesian(groups, df_all, line_extrap=0.15):
    big = [g for g in groups if len(g['points']) >= 60]
    plt.figure(figsize=(10,6))
    plt.scatter(df_all['theta'], df_all['r'], s=3, alpha=0.3, color='gray')
    colors = ['red','blue','green','purple','orange','brown','magenta','cyan','gold','gray','lime']
    for i, g in enumerate(big):
        c, pts = colors[i % len(colors)], g['points']
        pa_txt = f"{g['pa']:.2f}°" if g['pa'] is not None else "—"
        plt.scatter(pts['theta'], pts['r'], s=10, color=c,
                    label=f"G{i+1} ({len(pts)}) | PA={pa_txt}")
        if g['slope'] is not None:
            dt = g['theta_max'] - g['theta_min']
            t = np.linspace(g['theta_min']-line_extrap*dt, g['theta_max']+line_extrap*dt, 200)
            plt.plot(t, g['slope'] * t + g['intercept'], '--', color=c)
    plt.xlabel("θ (°)")
    plt.ylabel("r")
    plt.grid(True)
    plt.legend()

# ═══════════════════════════════════════════════════════════════════════
# 7-B) Polar plot for large groups
# ═══════════════════════════════════════════════════════════════════════
def plot_groups_polar(groups, df_all, line_extrap=0.15):
    plt.figure(figsize=(9,8))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(np.radians(df_all['theta']), df_all['r'], s=3, alpha=0.3, color='gray')
    colors = ['red','blue','green','purple','orange','brown','magenta','cyan','gold','gray','lime']
    for i, g in enumerate([g for g in groups if len(g['points']) >= 60]):
        c, pts = colors[i % len(colors)], g['points']
        pa_txt = f"{g['pa']:.2f}°" if g['pa'] is not None else "—"
        ax.scatter(np.radians(pts['theta']), pts['r'], s=10, color=c,
                   label=f"G{i+1} ({len(pts)}) | PA={pa_txt}")
        if g['slope'] is not None:
            dt = g['theta_max'] - g['theta_min']
            t_deg = np.linspace(g['theta_min'] - line_extrap * dt, g['theta_max'] + line_extrap * dt, 200)
            r_line = g['slope'] * t_deg + g['intercept']
            ax.plot(np.radians(t_deg), r_line, '.-', color='gray', alpha=0.20)
            ax.plot(np.radians(t_deg), r_line, '--', color=c)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

# ═══════════════════════════════════════════════════════════════════════
# 8) HistogramSegmenter and support dataclasses
# ═══════════════════════════════════════════════════════════════════════
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
    def __init__(self,
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
                 dr_multiplier: float = 3.0):
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

    def _histogram(self, pts: np.ndarray):
        hist, te, re = np.histogram2d(
            pts[:, 0], pts[:, 1],
            bins=[self.bins_theta, self.bins_r]
        )
        hist_s = gaussian_filter(hist, sigma=self.smooth_sigma)
        return hist_s, te, re

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

    def _clusters_and_skeletons(self, pts, find, labels, te, re):
        tidx = np.clip(np.digitize(pts[:, 0], te) - 1, 0, self.bins_theta - 1)
        ridx = np.clip(np.digitize(pts[:, 1], re) - 1, 0, self.bins_r - 1)
        merged = {}
        for p, (ti, ri) in enumerate(zip(tidx, ridx)):
            lab = labels[ti, ri]
            if lab:
                merged.setdefault(find(lab), []).append(pts[p])
        clusters = [np.vstack(v) for v in merged.values() if len(v) >= self.min_cluster_size]

        skeletons = []
        for cl in clusters:
            thetas, rs = cl[:, 0], cl[:, 1]
            bins_arr = np.arange(thetas.min(), thetas.max() + self.theta_bin_size, self.theta_bin_size)
            ske = []
            for k in range(len(bins_arr)-1):
                m = (thetas >= bins_arr[k]) & (thetas < bins_arr[k+1])
                if m.any():
                    ske.append((thetas[m].mean(), rs[m].mean()))
            skeletons.append(sorted(ske, key=lambda x: -x[0]))
        return clusters, skeletons

    def _density_path(self, a, b, hist_s, te, re):
        theta_res, r_res = te[1]-te[0], re[1]-re[0]
        steps = max(int(abs(b[0]-a[0]) / self.theta_step)+1, 2)
        thetas = np.linspace(a[0], b[0], steps)
        rs_seg = np.linspace(a[1], b[1], steps)
        radius = math.hypot(b[0]-a[0], b[1]-a[1]) * self.density_ratio

        out = []
        for θ_i, r_i in zip(thetas, rs_seg):
            ti = int(np.clip(np.digitize(θ_i, te) - 1, 0, hist_s.shape[0]-1))
            ri0 = int(np.clip(np.digitize(r_i, re) - 1, 0, hist_s.shape[1]-1))
            max_off = int(math.ceil(radius / r_res))
            vals, offs = [], []
            for off in range(-max_off, max_off+1):
                idx = ri0 + off
                if 0 <= idx < hist_s.shape[1]:
                    vals.append(hist_s[ti, idx]); offs.append(off)
            for k in range(1, len(vals)-1):
                if vals[k] > vals[k-1] and vals[k] > vals[k+1] and vals[k] > 0:
                    rc = re[ri0+offs[k]] + r_res/2.0
                    if abs(rc-r_i) <= self.r_threshold:
                        out.append((θ_i, rc))
        return out

    def _candidate_segments(self, skeletons):
        segs = []
        for ske in skeletons:
            segs.extend(list(zip(ske, ske[1:])))
        for i, ske in enumerate(skeletons):
            if not ske: continue
            a = ske[-1]
            best, bd = None, math.inf
            for j, ske2 in enumerate(skeletons):
                if i == j or not ske2: continue
                b = ske2[0]
                dr, dθ = b[1]-a[1], b[0]-a[0]
                if dr>0 and dθ<0:
                    d = math.hypot(dr, dθ)
                    if d<bd:
                        bd, best = d, (a,b)
            if best: segs.append(best)
        return segs

    def _build_contours(self, segs, hist_s, te, re):
        contours, connections = [], []
        for a,b in segs:
            path = self._density_path(a, b, hist_s, te, re)
            route = [a] + path + [b]
            contours.append(route)
            connections.append(Connection(a, b, abs(b[1]-a[1]), math.hypot(b[0]-a[0], b[1]-a[1])))
        return contours, connections

    @staticmethod
    def _split_by_threshold(contours, thr):
        out = []
        for cnt in contours:
            tmp = [cnt[0]]
            for a, b in zip(cnt, cnt[1:]):
                if abs(b[1]-a[1]) <= thr:
                    tmp.append(b)
                else:
                    if len(tmp)>1: out.append(tmp)
                    tmp = [b]
            if len(tmp)>1: out.append(tmp)
        return out

    def run(self, datos: Dict[str, Any]):
        bg = datos['background'][['theta','r']].values
        grp = np.vstack([g['points'][['theta','r']].values for g in datos['grupos_finales']])
        pts = np.vstack([bg, grp])

        hist_s, te, re = self._histogram(pts)
        find, labels = self._weighted_bfs(hist_s, te, re)
        clusters, skeletons = self._clusters_and_skeletons(pts, find, labels, te, re)
        segs = self._candidate_segments(skeletons)
        raw_contours, conns = self._build_contours(segs, hist_s, te, re)
        mean_dr = float(np.mean([c.delta_r for c in conns])) if conns else 0.0
        thr = mean_dr * self.dr_multiplier
        final_contours = self._split_by_threshold(raw_contours, thr)

        objects = [IslandObject('isla-conexion-isla', cnt) for cnt in final_contours]
        for c in conns:
            if c.delta_r > thr:
                objects.append(IslandObject('isla-conexiones-adicionales', [c.a,c.b]))

        return {
            'hist_s': hist_s, 'te': te, 're': re,
            'clusters': clusters, 'skeletons': skeletons,
            'contours': final_contours, 'connections': conns,
            'mean_dr': mean_dr, 'threshold': thr, 'objects': objects
        }

    def quick_plot(self, result, figsize=(14,6)):
        plt.figure(figsize=figsize)
        plt.imshow(result['hist_s'].T, origin='lower', extent=[result['te'][0], result['te'][-1], result['re'][0], result['re'][-1]], aspect='auto', cmap='inferno')
        plt.colorbar(label='Densidad suavizada')
        for idx, cl in enumerate(result['clusters']):
            plt.scatter(cl[:,0], cl[:,1], s=8, alpha=0.6, label=f'Isla {idx}')
        for ske in result['skeletons']:
            arr = np.array(ske)
            if arr.size:
                plt.scatter(arr[:,0], arr[:,1], c='w', s=40, edgecolors='k')
        for cnt in result['contours']:
            xs, ys = zip(*cnt)
            plt.plot(xs, ys, '-o', color='cyan', markersize=5)
        # no show here; saving externally

# ═══════════════════════════════════════════════════════════════════════
# 9) Main pipeline per ID and argument parsing
# ═══════════════════════════════════════════════════════════════════════
def process_halo(id_halo: str, args):
    print(f"Processing ID {id_halo}...")
    bfs_clusters, df_f = generate_bfs_seeds(
        id_halo=id_halo,
        theta_min=args.theta_min, theta_max=args.theta_max,
        quartile_threshold=args.quartile_threshold,
        theta_diff=args.theta_diff, r_diff=args.r_diff,
        gap_threshold_theta=args.gap_theta, gap_threshold_r=args.gap_r,
        file_prefix=args.file_prefix
    )
    print(f"  BFS seeds: {len(bfs_clusters)} clusters")
    merged = adjust_and_merge_seeds(bfs_clusters, args.slope_variation, args.bounding_extrap)
    print(f"  After merge: {len(merged)} groups")
    validated = validate_dispersion_and_reprocess(merged, args.dispersion_threshold, args.reproc_theta, args.reproc_r)
    print(f"  After validate: {len(validated)} groups")

    # Save Cartesian plot
    plot_groups_cartesian(validated, df_f, args.line_extrap)
    fig = plt.gcf()
    out_cart = os.path.join(args.output_dir, f"{id_halo}_cartesian.png")
    fig.savefig(out_cart)
    plt.close(fig)

    # Save Polar plot
    plot_groups_polar(validated, df_f, args.line_extrap)
    fig = plt.gcf()
    out_polar = os.path.join(args.output_dir, f"{id_halo}_polar.png")
    fig.savefig(out_polar)
    plt.close(fig)

    # Prepare data for segmentation
    grupos_finales = [g for g in validated if len(g['points']) >= args.min_points]
    puntos_grupos = set()
    for g in grupos_finales:
        puntos_grupos.update(g['points']['id'].tolist())
    background = df_f[~df_f['id'].isin(puntos_grupos)].copy()

    datos = {'grupos_finales': grupos_finales, 'background': background}
    segmenter = HistogramSegmenter(
        bins_theta=args.bins_theta,
        bins_r=args.bins_r,
        smooth_sigma=args.smooth_sigma,
        theta_bin_size=args.theta_bin_size,
        dr_multiplier=args.dr_multiplier
    )
    result = segmenter.run(datos)

    # Save histogram segmentation quick plot
    segmenter.quick_plot(result)
    fig = plt.gcf()
    out_hist = os.path.join(args.output_dir, f"{id_halo}_histogram.png")
    fig.savefig(out_hist)
    plt.close(fig)

    print(f"  Saved plots: {out_cart}, {out_polar}, {out_hist}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline for halo segmentation and histogram analysis")
    parser.add_argument('--ids', nargs='+', required=True, help="List of halo IDs to process")
    parser.add_argument('--file-prefix', default='data_rho', help="Prefix for input CSV files")
    parser.add_argument('--output-dir', default='results', help="Directory to save output images")

    # Data filtering
    parser.add_argument('--theta-min', type=float, default=0.0)
    parser.add_argument('--theta-max', type=float, default=360.0)
    parser.add_argument('--quartile-threshold', type=float, default=0.525)
    parser.add_argument('--theta-diff', type=float, default=3.0)
    parser.add_argument('--r-diff', type=float, default=0.5)
    parser.add_argument('--gap-theta', type=float, default=1.80)
    parser.add_argument('--gap-r', type=float, default=1.0)

    # Merge & validation
    parser.add_argument('--slope-variation', type=float, default=0.30)
    parser.add_argument('--bounding-extrap', type=float, default=0.35)
    parser.add_argument('--dispersion-threshold', type=float, default=1.80)
    parser.add_argument('--reproc-theta', type=float, default=1.25)
    parser.add_argument('--reproc-r', type=float, default=0.4)
    parser.add_argument('--min-points', type=int, default=60)

    # Plotting
    parser.add_argument('--line-extrap', type=float, default=0.15)

    # HistogramSeg
    parser.add_argument('--bins-theta', type=int, default=120)
    parser.add_argument('--bins-r', type=int, default=80)
    parser.add_argument('--smooth-sigma', type=float, default=1.0)
    parser.add_argument('--theta-bin-size', type=float, default=8)
    parser.add_argument('--dr-multiplier', type=float, default=2.5)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for hid in args.ids:
        process_halo(hid, args)

if __name__ == '__main__':
    main()
