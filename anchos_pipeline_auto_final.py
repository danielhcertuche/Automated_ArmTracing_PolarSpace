#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anchos_pipeline.py
==================

"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd

# Matplotlib: backend configurable en main() según --no-show
import matplotlib
import matplotlib.pyplot as plt

# SciPy (solo se importa interpolate por compatibilidad)
from scipy import interpolate  # noqa: F401

# lmfit
from lmfit.models import SkewedGaussianModel

# shapely (no estrictamente requerido en la ruta principal pero solicitado)
try:
    from shapely.geometry import LineString, MultiPoint, Point  # noqa: F401
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("anchos_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

FWHM_CONST = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_halo_data(id_halo: str) -> pd.DataFrame:
    """Carga el CSV del halo filtrado.

    Debe contener columnas: x, y, deltaRho (al menos).
    """
    path = Path(f"data_rho_{id_halo}_filtered.csv")
    df = pd.read_csv(path)
    return df


def load_tracing_points_polar(id_halo: str) -> pd.DataFrame:
    """Carga skeletons_union y clusters; mapea brazos, filtra clusters
    con solapamiento ≤90 % en r y construye coordenadas cartesianas + id_point.

    Retorna DataFrame con columnas: theta, r, arm, cluster?, cluster_id?,
    x, y, id_point.
    """
    skel_path   = Path(f"TracingPoints/skeletons_union_{id_halo}.csv")
    clust_path  = Path(f"TracingPoints/clusters_{id_halo}.csv")
    df          = pd.read_csv(skel_path)
    df_clusters = pd.read_csv(clust_path)

    # Mapear brazos por skeleton_id
    df['arm'] = df['skeleton_id'].map({0: 'arm1', 1: 'arm2'})

    # Identificar clúster principal y filtrar solapamiento >90% en r
    counts     = df_clusters['cluster_id'].value_counts()
    if counts.empty:
        df['cluster'] = np.nan
    else:
        primary_id = counts.index[0]
        prim_rads  = set(df_clusters.loc[df_clusters['cluster_id'] == primary_id, 'r'])

        valid_ids = [primary_id]
        for cid in counts.index[1:]:
            if len(valid_ids) >= 2:
                break
            rads    = set(df_clusters.loc[df_clusters['cluster_id'] == cid, 'r'])
            overlap = len(rads & prim_rads) / len(rads) if rads else 0
            if overlap <= 0.9:
                valid_ids.append(cid)

        df_clusters = df_clusters[df_clusters['cluster_id'].isin(valid_ids)].copy()
        ordered = df_clusters['cluster_id'].value_counts().index.tolist()
        label_map = {ordered[i]: f'cluster{i+1}' for i in range(len(ordered))}
        df_clusters['cluster'] = df_clusters['cluster_id'].map(label_map)

        # Unir clusters filtrados, conservando todos los puntos skeleton
        df = pd.merge(
            df,
            df_clusters[['theta', 'r', 'cluster', 'cluster_id']],
            on=['theta', 'r'],
            how='outer'
        )

    # Coordenadas cartesianas
    theta_rad = np.deg2rad(df['theta'])
    df['x']   = df['r'] * np.cos(theta_rad)
    df['y']   = df['r'] * np.sin(theta_rad)

    # id_point secuencial por brazo (orden angular)
    df['theta_rad'] = theta_rad
    df = df.sort_values(['arm','theta_rad']).reset_index(drop=True)
    df['id_point'] = df.groupby('arm').cumcount() + 1
    df = df.drop(columns=['theta_rad'])
    return df


# ---------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------
def rotate_coords(x, y, angle):
    """Rota coordenadas 2D por *angle* radianes (convención x' hacia delante)."""
    c, s = np.cos(angle), np.sin(angle)
    return (c*x + s*y, -s*x + c*y)


# ---------------------------------------------------------------------
# Binning de máximos
# ---------------------------------------------------------------------
def compute_peak_maxima(x_band, y_band, xmin, xmax, bin_width):
    """Máximo por bin en rango [xmin, xmax].

    Devuelve arrays (xs_max, ys_max) solo donde el valor máximo en el bin
    es > 0. Esto replica la heurística del código de prueba.
    """
    x_band = np.asarray(x_band)
    y_band = np.asarray(y_band)
    mask = (x_band >= xmin) & (x_band <= xmax)
    xs = x_band[mask]
    ys = y_band[mask]
    if xs.size == 0:
        return np.array([]), np.array([])

    steps = int(np.ceil((xmax-xmin)/bin_width)) if bin_width > 0 else 1
    xs_max, ys_max = [], []
    for i in range(steps):
        lo, hi = xmin + i*bin_width, xmin + (i+1)*bin_width
        m = (xs >= lo) & (xs < hi) if i < steps-1 else (xs >= lo) & (xs <= hi)
        if np.any(m):
            xs_max.append(xs[m].max())
            ys_max.append(ys[m].max())
    xs_max = np.asarray(xs_max)
    ys_max = np.asarray(ys_max)
    maskp = ys_max > 0
    return xs_max[maskp], ys_max[maskp]


# ---------------------------------------------------------------------
# Segmentación gap+drop y fit (versión extensa del código de prueba)
# ---------------------------------------------------------------------
def analyze_profile_gapdrop_fit(x_vals,
                                y_vals,
                                *,
                                start_gt=0.0,
                                gap_thr=0.5,
                                coarse_dx=0.5,
                                coarse_n=None,
                                coarse_edges=None,
                                coarse_agg="max",
                                min_pts=1,
                                smooth=3,
                                drop_select_frac=0.60,
                                peak_prominence_min=0.0,
                                do_fit=True,
                                fit_pad=0.0,
                                fit_use_smooth=False,
                                fig=None,
                                ax=None,
                                title=None,
                                show=False,
                                save_path=None):
    """Analiza perfil Δρ(x) → detecta caídas pico→valle y ajusta SkewedGaussian.

    Esta función es una traslación casi literal de la versión *prueba_* con
    algunas salvaguardas menores y la opción *show* para controlar plt.show().
    """
    # ------------------------------------------------------------------
    # sanity
    # ------------------------------------------------------------------
    x = np.asarray(x_vals, dtype=float).ravel()
    y = np.asarray(y_vals, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("x_vals/y_vals vacíos.")

    # ordenar por x
    isort = np.argsort(x)
    x = x[isort]; y = y[isort]

    # filtrar x > start_gt
    mpos = x > start_gt
    if not np.any(mpos):
        raise ValueError(f"No hay datos con x > {start_gt}.")
    x_pos = x[mpos]; y_pos = y[mpos]

    # ------------------------------------------------------------------
    # Coarse binning
    # ------------------------------------------------------------------
    def _agg_y(yarr, mode):
        if mode == "max":   return np.max(yarr)
        if mode == "mean":  return np.mean(yarr)
        if mode == "median":return np.median(yarr)
        if mode == "p90":   return np.percentile(yarr, 90)
        if mode == "p95":   return np.percentile(yarr, 95)
        if mode == "p75":   return np.percentile(yarr, 75)
        if mode == "first": return yarr[0]
        if mode == "last":  return yarr[-1]
        return np.max(yarr)

    if coarse_edges is not None:
        edges = np.asarray(coarse_edges, dtype=float)
        edges = np.sort(edges)
        if edges[0] > x_pos[0]:
            edges = np.r_[x_pos[0], edges]
        if edges[-1] < x_pos[-1]:
            edges = np.r_[edges, x_pos[-1]]
    elif coarse_dx is not None and coarse_dx > 0:
        nb = int(np.ceil((x_pos[-1] - x_pos[0]) / coarse_dx))
        edges = x_pos[0] + np.arange(nb+1)*coarse_dx
        if edges[-1] < x_pos[-1]:
            edges = np.r_[edges, x_pos[-1]]
    elif coarse_n is not None and coarse_n > 1:
        edges = np.linspace(x_pos[0], x_pos[-1], int(coarse_n)+1)
    else:
        edges = None

    if edges is None:
        coarse_x = x_pos.copy()
        coarse_y = y_pos.copy()
    else:
        cx, cy = [], []
        for i in range(len(edges)-1):
            lo, hi = edges[i], edges[i+1]
            m = (x_pos >= lo) & (x_pos < hi) if i < len(edges)-2 else (x_pos >= lo) & (x_pos <= hi)
            if np.count_nonzero(m) < min_pts:
                continue
            cx.append(np.median(x_pos[m]))
            cy.append(_agg_y(y_pos[m], coarse_agg))
        coarse_x = np.asarray(cx)
        coarse_y = np.asarray(cy)
        if coarse_x.size == 0:
            coarse_x = x_pos.copy(); coarse_y = y_pos.copy()

    # suavizado
    if smooth > 1 and coarse_y.size >= smooth:
        k = int(smooth)
        ker = np.ones(k)/k
        coarse_y_s = np.convolve(coarse_y, ker, mode='same')
    else:
        coarse_y_s = coarse_y

    # ------------------------------------------------------------------
    # detect picos y valles
    # ------------------------------------------------------------------
    y_s = coarse_y_s
    n = y_s.size
    peaks_idx = []
    valleys_idx = []
    for i in range(1, n-1):
        if y_s[i] >= y_s[i-1] and y_s[i] > y_s[i+1]:
            peaks_idx.append(i)
        if y_s[i] <= y_s[i-1] and y_s[i] < y_s[i+1]:
            valleys_idx.append(i)
    if 0 not in peaks_idx and 0 not in valleys_idx:
        if n>1 and y_s[1] > y_s[0]:
            valleys_idx = [0] + valleys_idx
        else:
            peaks_idx   = [0] + peaks_idx
    if (n-1) not in peaks_idx and (n-1) not in valleys_idx:
        if n>1 and y_s[-1] < y_s[-2]:
            valleys_idx.append(n-1)
        else:
            peaks_idx.append(n-1)
    peaks_idx   = np.unique(peaks_idx)
    valleys_idx = np.unique(valleys_idx)

    drops = []
    for p_i in peaks_idx:
        if y_s[p_i] < peak_prominence_min:
            continue
        next_peaks = peaks_idx[peaks_idx > p_i]
        if next_peaks.size:
            p_next = next_peaks[0]
            sl = slice(p_i+1, p_next+1)
        else:
            sl = slice(p_i+1, n)
        if sl.start >= n:
            continue
        y_seg = y_s[sl]
        if y_seg.size == 0:
            continue
        v_local = int(np.argmin(y_seg))
        v_i = sl.start + v_local
        drop_amp = y_s[p_i] - y_s[v_i]
        drop_frac = drop_amp / y_s[p_i] if y_s[p_i] > 0 else np.nan
        drops.append({
            "idx_peak": p_i,
            "idx_valley": v_i,
            "x_peak": coarse_x[p_i],
            "y_peak": y_s[p_i],
            "x_valley": coarse_x[v_i],
            "y_valley": y_s[v_i],
            "drop_amp": drop_amp,
            "drop_frac": drop_frac,
        })
    if drops:
        drops_df = pd.DataFrame(drops).sort_values("x_peak", ignore_index=True)
    else:
        drops_df = pd.DataFrame(columns=["idx_peak","idx_valley","x_peak","y_peak","x_valley","y_valley","drop_amp","drop_frac"])

    # seleccionar drop
    if not drops_df.empty:
        sel_mask = drops_df["drop_frac"] >= drop_select_frac
        if sel_mask.any():
            sel_idx = int(drops_df[sel_mask].index[0])
        else:
            sel_idx = int(drops_df["drop_amp"].idxmax())
        selected_drop = drops_df.loc[sel_idx].to_dict()
    else:
        selected_drop = None

    # segmento y datos para fit
    if selected_drop is not None:
        x_stop = selected_drop["x_valley"]
    else:
        x_stop = coarse_x[-1]
    x_fit_min = coarse_x[0]
    x_fit_max = x_stop + fit_pad

    mseg = (x_pos >= x_fit_min) & (x_pos <= x_fit_max)
    segment_x = x_pos[mseg]
    segment_y = y_pos[mseg]

    mcoarse_seg = (coarse_x >= x_fit_min) & (coarse_x <= x_fit_max)
    coarse_x_seg = coarse_x[mcoarse_seg]
    coarse_y_seg = (coarse_y_s if fit_use_smooth else coarse_y)[mcoarse_seg]

    # fit
    fit_result = None
    fit_params = {}
    if do_fit and coarse_x_seg.size >= 3:
        model = SkewedGaussianModel(prefix='sg_')
        params = model.make_params()
        params['sg_center'].set(value=0.0, min=coarse_x_seg.min(), max=coarse_x_seg.max())
        params['sg_amplitude'].set(value=np.nanmax(coarse_y_seg), min=0)
        params['sg_sigma'].set(value=(coarse_x_seg.max()-coarse_x_seg.min())/4, min=0)
        params['sg_gamma'].set(value=0.0)

        fit_result = model.fit(coarse_y_seg, params, x=coarse_x_seg)

        p = fit_result.params
        amp_fit   = p['sg_amplitude'].value
        mean_fit  = p['sg_center'].value
        sigma_fit = p['sg_sigma'].value
        gamma_fit = p['sg_gamma'].value
        amp_err   = p['sg_amplitude'].stderr
        mean_err  = p['sg_center'].stderr
        sigma_err = p['sg_sigma'].stderr
        gamma_err = p['sg_gamma'].stderr

        FWHM = FWHM_CONST * sigma_fit
        FWHM_err = FWHM_CONST * sigma_err if sigma_err is not None else np.nan

        x_model = np.linspace(coarse_x_seg.min(), coarse_x_seg.max(), 500)
        y_model = fit_result.eval(x=x_model)

        half = amp_fit/2.0 if amp_fit is not None else np.nan
        if np.isfinite(half):
            above = y_model >= half
            if above.any():
                iL = np.argmax(above)
                iR = len(above) - np.argmax(above[::-1]) - 1
                w1 = x_model[iL]; w2 = x_model[iR]; w_sum = w1 + w2
            else:
                w1 = w2 = w_sum = np.nan
        else:
            w1 = w2 = w_sum = np.nan

        fit_params = {
            "amp_fit": amp_fit, "amp_err": amp_err,
            "mean_fit": mean_fit, "mean_err": mean_err,
            "sigma_fit": sigma_fit, "sigma_err": sigma_err,
            "gamma_fit": gamma_fit, "gamma_err": gamma_err,
            "FWHM": FWHM, "FWHM_err": FWHM_err,
            "w1": w1, "w2": w2, "w_sum": w_sum,
            "x_model": x_model, "y_model": y_model,
        }

    # ------------------------------------------------------------------
    # Figura diagnóstica
    # ------------------------------------------------------------------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x, y, '.', ms=2, color='0.8', label='Todos')
    ax.plot(x_pos, y_pos, '.', ms=3, color='C0', label=f'x>start_gt ({start_gt})')
    ax.plot(coarse_x, coarse_y, '-o', ms=4, color='C1', label='Coarse')
    if smooth > 1:
        ax.plot(coarse_x, coarse_y_s, '-', lw=1.5, color='C3', label=f'Suavizado (k={smooth})')
    ax.axhline(0.0, color='k', lw=0.8, ls='--')

    dx_coarse = np.diff(coarse_x)
    gap_idx = np.where(dx_coarse > gap_thr)[0]
    for gi in gap_idx:
        ax.axvline(coarse_x[gi+1], color='magenta', ls=':', lw=1)

    if drops:
        ax.scatter(coarse_x[[d["idx_peak"] for d in drops]],
                   y_s[[d["idx_peak"] for d in drops]],
                   marker='^', s=50, color='green', edgecolor='k', label='Picos')
        ax.scatter(coarse_x[[d["idx_valley"] for d in drops]],
                   y_s[[d["idx_valley"] for d in drops]],
                   marker='v', s=50, color='brown', edgecolor='k', label='Valles')

    if segment_x.size:
        ax.axvspan(segment_x.min(), segment_x.max(), color='C2', alpha=0.15, label='Segmento')

    if selected_drop is not None:
        ax.plot([selected_drop["x_peak"], selected_drop["x_valley"]],
                [selected_drop["y_peak"], selected_drop["y_valley"]],
                color='black', lw=2, label=f'Drop ≥{drop_select_frac:.0%}')
        ax.scatter(selected_drop["x_valley"], selected_drop["y_valley"],
                   marker='X', s=80, color='black', zorder=5)
        ax.axhline(selected_drop["y_peak"]*(1-drop_select_frac),
                   color='black', ls='--', lw=1,
                   label=f'Nivel {100*drop_select_frac:.0f}%')

    if do_fit and fit_params:
        ax.plot(fit_params["x_model"], fit_params["y_model"],
                'r-', lw=2, label='SkewedGaussian fit')

    if title is None:
        title = "Profile analysis (gap+drops+fit)"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Δρ")
    ax.legend(fontsize='small', ncol=2)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return {
        "drops_df": drops_df,
        "selected_drop": selected_drop,
        "segment_x": segment_x,
        "segment_y": segment_y,
        "coarse_x": coarse_x,
        "coarse_y": coarse_y,
        "coarse_y_s": coarse_y_s,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "x_all": x,
        "y_all": y,
        "fit_result": fit_result,
        "fit_params": fit_params,
        "fig": fig,
        "ax": ax,
    }


# ---------------------------------------------------------------------
# Función principal de cómputo por punto (usa análisis doble: +x y -x)
# ---------------------------------------------------------------------
def compute_point_detail_skewed_gausiano(id_halo: str,
                                         arm_label: str,
                                         n: int,
                                         *,
                                         b: float = 0.4,
                                         gap_threshold: float = 0.5,
                                         pad: float = 0.25,
                                         bin_width: float = 0.2,
                                         drop_select_frac: float = 0.60,
                                         show_debug: bool = False,
                                         debug_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Ejecuta todo el flujo de perfil local y ajuste Skewed Gaussian.

    Pasos:
      1. Carga datos halo + skeleton.
      2. Extrae perfil en banda |y'|<b tras rotar al eje radial local.
      3. Busca segmento continuo básico alrededor de x=0 (gaps > gap_threshold).
      4. Corre *analyze_profile_gapdrop_fit* hacia fuera (x>0) y hacia dentro (reflejado)
         para estimar límites por caída de densidad.
      5. Determina límites finales [limit_xneg, limit_xpos] combinando segmentación
         continua + caídas detectadas.
      6. Binning de máximos → datos para ajuste.
      7. Ajuste SkewedGaussian con pesos fuertes en extremos (0,0) forzados.
      8. Calcula w1, w2 a media altura del modelo ajustado.

    Devuelve dict con parámetros, errores, arrays y objeto fit_result.
    """
    # --- carga ---------------------------------------------------------
    df = load_halo_data(id_halo)
    df_skel = load_tracing_points_polar(id_halo)

    # seleccionar punto
    row_q = df_skel.query("arm == @arm_label and id_point == @n")
    if row_q.empty:
        raise ValueError(f"Punto no encontrado: halo={id_halo} {arm_label} n={n}")
    row = row_q.iloc[0]

    x0, y0 = row['x'], row['y']
    ang    = np.arctan2(y0, x0)

    xr_all, yr_all = rotate_coords(df['x'].values - x0,
                                   df['y'].values - y0, -ang)
    prop = df['deltaRho'].values

    # banda mid-plane
    mask = np.abs(yr_all) < b
    x_band = xr_all[mask]
    y_band = prop[mask]

    # ordenar
    order = np.argsort(x_band)
    x_band = x_band[order]; y_band = y_band[order]

    # segmento continuo simple (gaps en datos crudos)
    idx0 = np.argmin(np.abs(x_band)) if x_band.size else 0
    iL = iR = idx0
    while iL > 0 and (x_band[iL] - x_band[iL-1]) < gap_threshold:
        iL -= 1
    while iR < len(x_band)-1 and (x_band[iR+1] - x_band[iR]) < gap_threshold:
        iR += 1
    seg_x = x_band[iL:iR+1]; seg_y = y_band[iL:iR+1]

    max_seg = float(seg_x.max()) if seg_x.size else 0.0
    min_seg = float(seg_x.min()) if seg_x.size else 0.0

    # --- detección de valle (drop) hacia x>0 ---------------------------------
    try:
        dbg_path = None
        if show_debug and debug_dir is not None:
            dbg_path = debug_dir / f"dbg_fwd_{id_halo}_{arm_label}_{n:03d}.png"
        cl_seg_fwd = analyze_profile_gapdrop_fit(
            x_band, y_band,
            start_gt=0.1,
            gap_thr=gap_threshold,
            coarse_dx=gap_threshold,
            smooth=3,
            drop_select_frac=drop_select_frac,
            do_fit=True,
            show=show_debug,
            save_path=dbg_path,
        )
        sel_fwd = cl_seg_fwd["selected_drop"]
    except Exception as e:  # pragma: no cover - debug fallback
        logger.debug(f"analyze_profile_gapdrop_fit fwd falló: {e}")
        sel_fwd = None
        cl_seg_fwd = {"drops_df": pd.DataFrame()}

    if sel_fwd is not None:
        x_valley_pos = float(sel_fwd["x_valley"])
    else:
        df_drops = cl_seg_fwd.get("drops_df", pd.DataFrame())
        if not df_drops.empty:
            m = df_drops["drop_frac"] >= drop_select_frac
            if m.any():
                x_valley_pos = float(df_drops.loc[m].iloc[0]["x_valley"])
            else:
                x_valley_pos = float(df_drops.loc[df_drops["drop_amp"].idxmax()]["x_valley"])
        else:
            x_valley_pos = max_seg

    # --- detección hacia x<0 usando espejo -----------------------------------
    try:
        dbg_path = None
        if show_debug and debug_dir is not None:
            dbg_path = debug_dir / f"dbg_bwd_{id_halo}_{arm_label}_{n:03d}.png"
        cl_seg_bwd = analyze_profile_gapdrop_fit(
            -x_band, y_band,         # espejo horizontal
            start_gt=0.1,
            gap_thr=gap_threshold,
            coarse_dx=gap_threshold,
            smooth=3,
            drop_select_frac=drop_select_frac,
            do_fit=True,
            show=show_debug,
            save_path=dbg_path,
        )
        sel_bwd = cl_seg_bwd["selected_drop"]
    except Exception as e:  # pragma: no cover
        logger.debug(f"analyze_profile_gapdrop_fit bwd falló: {e}")
        sel_bwd = None
        cl_seg_bwd = {"drops_df": pd.DataFrame()}

    if sel_bwd is not None:
        x_valley_neg = -float(sel_bwd["x_valley"])
    else:
        df_drops = cl_seg_bwd.get("drops_df", pd.DataFrame())
        if not df_drops.empty:
            m = df_drops["drop_frac"] >= drop_select_frac
            if m.any():
                x_valley_neg = -float(df_drops.loc[m].iloc[0]["x_valley"])
            else:
                x_valley_neg = -float(df_drops.loc[df_drops["drop_amp"].idxmax()]["x_valley"])
        else:
            x_valley_neg = min_seg

    # límites finales combinando segmento crudo + valles drop
    limit_xpos = min(max_seg, x_valley_pos)
    limit_xneg = max(min_seg, x_valley_neg)

    # padding para ajuste
    fit_min = limit_xneg - pad
    fit_max = limit_xpos + pad

    # binning de máximos dentro de límites
    xs_max, ys_max = compute_peak_maxima(seg_x, seg_y, limit_xneg, limit_xpos, bin_width)

    # serie augmentada con ceros extremos pesados
    x_aug = np.concatenate(([limit_xneg, limit_xpos], xs_max))
    y_aug = np.concatenate(([0.0, 0.0], ys_max))
    w = np.ones_like(y_aug)
    if w.size >= 2:
        w[0:2] = 10.0

    # --- ajuste Skewed Gaussian ----------------------------------------------
    model = SkewedGaussianModel(prefix='sg_')
    params = model.make_params()
    params['sg_center'].set(value=0.0, min=fit_min, max=fit_max)
    params['sg_amplitude'].set(value=float(np.nanmax(y_aug)) if y_aug.size else 1.0, min=0.0)
    params['sg_sigma'].set(value=max((limit_xpos-limit_xneg)/4.0, 1e-6), min=0.0)
    params['sg_gamma'].set(value=0.0, min=-5.0, max=5.0, vary=True)

    if x_aug.size >= 3:
        result = model.fit(y_aug, params, x=x_aug, weights=w)
    else:
        # fallback: usa segmento crudo
        result = model.fit(seg_y, params, x=seg_x)

    p = result.params
    amp_fit     = p['sg_amplitude'].value
    amp_err     = p['sg_amplitude'].stderr
    mean_fit    = p['sg_center'].value
    mean_err    = p['sg_center'].stderr
    sigma_fit   = p['sg_sigma'].value
    sigma_err   = p['sg_sigma'].stderr
    gamma_fit   = p['sg_gamma'].value
    gamma_err   = p['sg_gamma'].stderr

    FWHM        = FWHM_CONST * sigma_fit
    FWHM_err    = FWHM_CONST * sigma_err if sigma_err is not None else np.nan

    # half-level widths
    xfit_dense = np.linspace(fit_min, fit_max, 1000)
    yfit_dense = result.eval(x=xfit_dense)
    level = amp_fit/2.0 if amp_fit is not None else np.nan

    idx_peak = np.argmax(yfit_dense)
    x_peak_fit = float(xfit_dense[idx_peak])
    y_peak_fit = float(yfit_dense[idx_peak])

    level = y_peak_fit/2.0 if y_peak_fit is not None else np.nan

    def _half_bounds(xa, ya, lvl):
        if not np.isfinite(lvl):
            return np.nan, np.nan
        # buscar cruces (sign change de ya-lvl)
        diff = ya - lvl
        sign = np.sign(diff)
        idx = np.where(sign[:-1]*sign[1:] < 0)[0]
        if idx.size >= 2:
            # primer y último cruce
            iL = idx[0]; iR = idx[-1]+1
            # interpolación lineal
            xL = np.interp(lvl, ya[iL:iL+2], xa[iL:iL+2])
            xR = np.interp(lvl, ya[iR-1:iR+1], xa[iR-1:iR+1])
            return xL, xR
        elif idx.size == 1:
            # solo un cruce; buscar max para otro lado
            i = idx[0]
            xC = xa[i]
            return xC, xC
        else:
            return np.nan, np.nan

    w1, w2 = _half_bounds(xfit_dense, yfit_dense, level)
    w_sum = (abs(w1) + abs(w2)) if np.all(np.isfinite([w1,w2])) else np.nan

    return {
        'id_halo': id_halo,
        'arm': arm_label,
        'id_point': n,
        'x0': x0, 'y0': y0,
        'b': b,
        'gap_threshold': gap_threshold,
        'pad': pad,
        'bin_width': bin_width,
        'limit_xneg': limit_xneg,
        'limit_xpos': limit_xpos,
        'fit_min': fit_min,
        'fit_max': fit_max,
        'amp_fit': amp_fit,     'amp_err': amp_err,
        'mean_fit': mean_fit,   'mean_err': mean_err,
        'sigma_fit': sigma_fit, 'sigma_err': sigma_err,
        'gamma_fit': gamma_fit, 'gamma_err': gamma_err,
        'FWHM': FWHM,           'FWHM_err': FWHM_err,
        'w1': w1, 'w2': w2, 'w_sum': w_sum,
        'x_band': x_band, 'y_band': y_band,
        'seg_x': seg_x, 'seg_y': seg_y,
        'xs_max': xs_max, 'ys_max': ys_max,
        'x_aug': x_aug, 'y_aug': y_aug,
        'weights': w,
        'fit_result': result,
    }


# ---------------------------------------------------------------------
# Plot de diagnóstico y guardado (PNG + CSV arrays)
# ---------------------------------------------------------------------
def plot_point_detail_skewed_gausian(data: Dict[str, Any],
                                     save_path: Path,
                                     *,
                                     show: bool = False,
                                     save_csv: bool = True) -> None:
    """Genera y guarda figura del perfil local y ajuste SkewedGaussian.

    Además (opcional) escribe un CSV hermano con curvas de modelo y datos binned.
    """
    res = data['fit_result']
    x_band = data['x_band']; y_band = data['y_band']
    seg_x  = data['seg_x'];  seg_y  = data['seg_y']
    xs_max = data['xs_max']; ys_max = data['ys_max']
    fit_min = data['fit_min']; fit_max = data['fit_max']
    amp_fit = data['amp_fit']
    w1 = data['w1']; w2 = data['w2']

    # modelo denso
    x_fit = np.linspace(fit_min, fit_max, 1000)
    y_fit = res.eval(x=x_fit)

    created_fig = True
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x_band, y_band, s=3, alpha=0.3, label='Perfil Δρ')
    ax.scatter(xs_max, ys_max, s=30, color='magenta', label='Máximos por bin')
    ax.plot(x_fit, y_fit, 'r-', lw=2, label='Skewed Gaussian Fit')
    ax.axvline(fit_min, ls='-.', color='olive', label=f'Fit min {fit_min:.2f}')
    ax.axvline(fit_max, ls='-.', color='olive', label=f'Fit max {fit_max:.2f}')
    if np.isfinite(w1):
        ax.axvline(w1, ls='--', color='black', label=f'w1 {w1:.2f}')
    if np.isfinite(w2):
        ax.axvline(w2, ls='--', color='black', label=f'w2 {w2:.2f}')

    # pico modelado
    idx_peak = np.argmax(y_fit)
    x_peak_fit = float(x_fit[idx_peak])
    y_peak_fit = float(y_fit[idx_peak])
    ax.scatter(x_peak_fit, y_peak_fit, s=80, marker='X', color='C1', label=f'peak_fit({y_peak_fit:.2f})')

    ax.set(xlabel='Distance [kpc]', ylabel='ΔDensity',
           title=f"Halo {data['id_halo']} • {data['arm']} • n={data['id_point']}")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if save_csv:
        # Curvas + binned
        outcsv = save_path.with_suffix('.csv')
        dfc = pd.DataFrame({
            'x_band': pd.Series(x_band),
            'y_band': pd.Series(y_band),
        })
        # extras (padding with NaNs lengths mismatch)
        df_fit = pd.DataFrame({'x_fit': x_fit, 'y_fit': y_fit})
        df_bin = pd.DataFrame({'xs_max': xs_max, 'ys_max': ys_max})
        # write multiple frames to one csv? easiest: to dict-of-dicts wide
        # We'll concat with keys-level
        dfc = pd.concat({
            'band': dfc,
            'fit': df_fit,
            'bin': df_bin,
        }, axis=1)
        dfc.to_csv(outcsv, index=False)


def plot_skeletons_single(id_halo: str,
                          save_path: Optional[Path] = None,
                          *,
                          show: bool = False,
                          alpha_bg: float = 0.20) -> None:
    """Grafica el halo completo con los dos brazos conectados y (si *save_path*) guarda PNG."""
    df_skel = load_tracing_points_polar(id_halo)
    df_halo = load_halo_data(id_halo)

    # Paleta
    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fondo halo
    ax.scatter(df_halo['x'], df_halo['y'], s=1, c='grey', alpha=alpha_bg, label='Background')

    # Clusters (si existen)
    if 'cluster' in df_skel.columns:
        for i, (cluster_label, grp) in enumerate(df_skel.dropna(subset=['cluster']).groupby('cluster')):
            col = palette[i % len(palette)]
            ax.scatter(grp['x'], grp['y'],
                       s=30, alpha=0.24,
                       edgecolor=col, linewidth=1)

    # Brazos conectados
    for j, (arm_label, grp) in enumerate(df_skel.groupby('arm')):
        col = palette[(j-2) % len(palette)]
        grp_sorted = grp.sort_values('id_point')
        ax.plot(grp_sorted['x'], grp_sorted['y'],
                linestyle='-', color='black',
                linewidth=1.5, alpha=0.42, zorder=2)
        ax.scatter(grp_sorted['x'], grp_sorted['y'],
                   marker='*', s=9, color=col,
                   linewidth=1.5, label=arm_label, zorder=3)

    ax.set_aspect('equal')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_title(f'Halo {id_halo}')
    leg = ax.legend()
    for handle in leg.legendHandles:
        try:
            handle.set_alpha(1)
        except Exception:
            pass

    # Guardar
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    # Mostrar o cerrar
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------------------------------------------------------------
# Procesamiento de halo completo
# ---------------------------------------------------------------------
def process_halo(id_halo: str,
                 *,
                 b: float = 0.4,
                 bin_width: float = 0.2,
                 output_dir: Path = Path('ANCHOS_OUT'),
                 gap_threshold: float = 0.5,
                 pad: float = 0.25,
                 no_show: bool = False) -> None:
    """Procesa todos los puntos de todos los brazos de un halo."""
    output_dir = Path(output_dir)
    figs_dir = output_dir / 'FIGS'
    fitp_dir = output_dir / 'FIT_PARAMS'
    dbg_dir  = output_dir / 'DEBUG'
    figs_dir.mkdir(parents=True, exist_ok=True)
    fitp_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    df_skel = load_tracing_points_polar(id_halo)
    logger.info(f"Halo {id_halo}: {len(df_skel)} puntos en {df_skel['arm'].nunique()} brazos.")

    try:
        plot_skeletons_single(
            id_halo,
            save_path=figs_dir / f"{id_halo}_skeletons.png",
            show=(not no_show)
        )
    except Exception as e:
        warnings.warn(f"No se pudo generar figura de skeletons para halo {id_halo}: {e}")

    all_rows = []
    for arm_label, grp in df_skel.groupby('arm'):
        logger.info(f"Procesando brazo {arm_label} ({len(grp)} puntos)...")
        for n in grp['id_point'].astype(int):
            try:
                data = compute_point_detail_skewed_gausiano(
                    id_halo, arm_label, n,
                    b=b,
                    gap_threshold=gap_threshold,
                    pad=pad,
                    bin_width=bin_width,
                    show_debug=False,
                    debug_dir=dbg_dir,
                )
            except Exception as e:
                warnings.warn(f"Fallo en halo={id_halo} {arm_label} n={n}: {e}")
                continue

            # guardar figura & csv de curvas
            fig_path = figs_dir / f"{id_halo}_{arm_label}_{n:03d}.png"
            plot_point_detail_skewed_gausian(
                data, fig_path, show=(not no_show), save_csv=False
            )

            # CSV de parámetros (uno por punto)
            row = {
                k: data[k] for k in [
                    'id_halo','arm','id_point','x0','y0','b',
                    'limit_xneg','limit_xpos','fit_min','fit_max',
                    'amp_fit','amp_err','mean_fit','mean_err',
                    'sigma_fit','sigma_err','gamma_fit','gamma_err',
                    'FWHM','FWHM_err','w1','w2','w_sum',
                ]
            }
            all_rows.append(row)
            pd.DataFrame([row]).to_csv(
                fitp_dir / f"{id_halo}_{arm_label}_{n:03d}.csv", index=False
            )

    # resumen global
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(output_dir / f"anchos_{id_halo}_ALL.csv", index=False)
        logger.info(f"Listo. {len(df_all)} puntos procesados.")
    else:
        logger.warning("No se procesó ningún punto.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Pipeline de anchos de brazo (SkewedGaussian).")

    ap.add_argument("id_halo", help="Identificador del halo, p.ej. 11")

    ap.add_argument("-b", "--band", type=float, default=0.4,
                    help="Semiancho de la banda |y'|<b [kpc].")

    ap.add_argument("-w", "--bin-width", type=float, default=0.2,
                    help="Ancho de bin para máximos [kpc].")

    ap.add_argument("-o", "--out", default="results_demo",
                    help="Directorio de salida.")

    ap.add_argument("--gap", type=float, default=0.5,
                    help="Umbral de gap [kpc] para segmentación continua.")

    ap.add_argument("--pad", type=float, default=0.25,
                    help="Padding lateral añadido al rango de ajuste [kpc].")

    ap.add_argument("--no-show", action="store_true",
                    help="No mostrar figuras en pantalla (modo batch).")

    return ap


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    # backend matplotlib para modo batch
    if args.no_show:
        try:
            matplotlib.use("Agg")
        except Exception:  # pragma: no cover
            pass

    process_halo(
        args.id_halo,
        b=args.band,
        bin_width=args.bin_width,
        output_dir=Path(args.out),
        gap_threshold=args.gap,
        pad=args.pad,
        no_show=args.no_show,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
