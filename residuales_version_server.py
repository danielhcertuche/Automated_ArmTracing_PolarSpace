#!/usr/bin/env python3
# coding: utf-8
"""
Procesador batch optimizado TNG50 – ajustes logarítmicos + plano XY (2025-06-19)
-------------------------------------------------------------------------------
* Lee CSV de un directorio de entrada usando IDs.
* Guarda figuras (.png) con .savefig().
* Calcula spline (lineal) del perfil radial (ρ̄) con **Rmax dinámico = 95 %** del radio máximo.
* Ajuste log-lineal por tramos y residuales (rho_resta1-4 y rho_resta_final).
* CSV resumido y completo.
* Figuras: perfil ρ vs R, Δρ/ρ vs R y **plano XY coloreado por deltaRho**.
"""
import os
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt

# Suprimir avisos innecesarios de pandas
pd.options.mode.chained_assignment = None


def log(msg: str) -> None:
    """Imprime mensaje con timestamp UTC."""
    print(f"[{datetime.utcnow().isoformat()} UTC] {msg}")


def load_data(file_path: str, chunks: int = 9) -> np.ndarray:
    """Lee un CSV grande por chunks y devuelve un ndarray."""
    log(f"Iniciando lectura de datos desde '{file_path}' en {chunks} trozos")
    cols_required = [
        "x", "y", "z", "vx", "vy", "vz",
        "lxvel", "lyvel", "lzvel", "Potential", "U",
        "rho", "index_id",
    ]
    total = sum(1 for _ in open(file_path)) - 1
    chunk_size = max(1, total // chunks)
    reader = pd.read_csv(file_path, usecols=cols_required, chunksize=chunk_size)

    arrays = [chunk.values for chunk in reader]
    if not arrays:
        raise ValueError("Archivo vacío o sin columnas requeridas")

    data = np.vstack(arrays)
    log(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
    return data


def save_fig(fig, path: str) -> None:
    """Guarda y cierra una figura."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Figura guardada: {path}")


def process_halo(halo_id: str, in_dir: str, out_dir: str) -> None:
    """Procesa un halo: contraste de densidades y visualizaciones."""
    t0 = time.time()
    log(f"=== Iniciando procesamiento del halo {halo_id} ===")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- carga de datos ----------
    df = pd.DataFrame(
        load_data(os.path.join(in_dir, f"halo_{halo_id}_datos_dbscan_fu.csv")),
        columns=[
            "x", "y", "z", "vx", "vy", "vz",
            "lxvel", "lyvel", "lzvel", "Potential", "U",
            "rho", "index_id",
        ],
    )

    # ---------- coordenadas y filtro disco fino ----------
    df["Rs"] = np.sqrt(df.x**2 + df.y**2)
    df["Zs"] = df.z
    df_sorted = df.sort_values("Rs")
    df_disk = df_sorted[(df_sorted.Zs.abs() <= 1.5) & (df_sorted.Rs > 1.0)]
    log(f"Filas tras filtrado espacial: {len(df_disk)} / {len(df_sorted)}")

    # ---------- perfil radial medio ----------
    deltaR = 0.1
    Rmax = np.floor(0.95 * df_disk.Rs.max() / deltaR) * deltaR
    Rs_mean, rho_mean = [], []
    step = 0.0
    while step + deltaR <= Rmax:
        mask = (df_disk.Rs > step) & (df_disk.Rs <= step + deltaR)
        if mask.any():
            Rs_mean.append(round((df_disk.Rs[mask].min() + df_disk.Rs[mask].max()) * 0.5, 2))
            rho_mean.append(df_disk.rho[mask].mean())
        step += deltaR
    Rs_mean, rho_mean = np.array(Rs_mean), np.array(rho_mean)
    ylog = np.log10(rho_mean)

    # spline lineal (para contraste)
    spline = interpolate.interp1d(Rs_mean, rho_mean, kind="linear", fill_value="extrapolate")

    # ---------- contraste deltaRho ----------
    df_lim = df_disk[(df_disk.Rs > Rs_mean.min()) & (df_disk.Rs < Rs_mean.max())].copy()
    df_lim["rho_sph_log"] = np.log10(df_lim.rho)
    df_lim["deltaRho"] = df_lim.rho / spline(df_lim.Rs) - 1.0
    filt = df_lim[(df_lim.deltaRho > 0) & (df_lim.deltaRho < Rmax)].copy()

    # ---------- ajuste log-lineal por tramos ----------
    cuts = [3.0, 5.0, 10.75, Rs_mean.max()]
    def line(x, a, b):
        return a * x + b

    params = []
    for i, hi in enumerate(cuts):
        lo = cuts[i - 1] if i else -np.inf
        seg = (Rs_mean > lo) & (Rs_mean <= hi)
        if seg.sum() >= 2:
            popt, _ = optimize.curve_fit(line, Rs_mean[seg], ylog[seg])
        else:
            popt = np.array([0.0, 0.0])
        params.append(popt)

    for idx, (a, b) in enumerate(params, 1):
        filt[f"rho_resta{idx}"] = filt.rho_sph_log - (a * filt.Rs + b) + filt.rho_sph_log.min()

    conds = [
        filt.Rs < cuts[0],
        (filt.Rs >= cuts[0]) & (filt.Rs < cuts[1]),
        (filt.Rs >= cuts[1]) & (filt.Rs <= cuts[2]),
        (filt.Rs >= cuts[2]) & (filt.Rs <= cuts[3]),
    ]
    choices = [filt[f"rho_resta{i}"] for i in range(1, 5)]
    filt["rho_resta_final"] = np.select(conds, choices)
    filt["rho_resta_final_exp"] = 10 ** filt.rho_resta_final

    # ---------- guardar CSVs ----------
    filt[["x", "y", "z", "Rs", "vx", "vy", "vz", "rho_resta_final_exp"]].to_csv(
        os.path.join(out_dir, f"data_rho_{halo_id}.csv"), index=False
    )

    full_cols = [
        "x", "y", "z", "vx", "vy", "vz", "lxvel", "lyvel", "lzvel",
        "Potential", "U", "rho", "Rs", "Zs", "rho_sph_log", "deltaRho",
    ] + [f"rho_resta{i}" for i in range(1, 5)] + ["rho_resta_final", "rho_resta_final_exp", "index_id"]
    filt[full_cols].to_csv(
        os.path.join(out_dir, f"data_rho_{halo_id}_filtered.csv"), index=False
    )

    # ---------- figuras ----------
    # perfil ρ vs R
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_sorted.Rs, df_sorted.rho, s=0.3, alpha=0.45, label="Data")
    ax.scatter(Rs_mean, rho_mean, facecolors="none", edgecolors="k", marker="v", label="Mean ρ")
    xprof = np.linspace(Rs_mean.min(), Rs_mean.max(), 2000)
    ax.plot(xprof, spline(xprof), lw=1.4, label="Spline (lineal)")
    ax.set(title=f"Halo {halo_id}: ρ vs R", xlabel="R [kpc]", ylabel="ρ", yscale="log")
    ax.grid(alpha=0.3); ax.legend()
    save_fig(fig, os.path.join(out_dir, f"rho_vs_R_{halo_id}.png"))

    # deltaRho vs R
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df_lim.Rs, df_lim.deltaRho, s=0.45, c=df_lim.deltaRho, cmap="plasma")
    ax.set(title=f"Halo {halo_id}: Δρ contraste", xlabel="R [kpc]", ylabel="deltaRho")
    ax.grid(alpha=0.3); fig.colorbar(sc, ax=ax, label="deltaRho")
    save_fig(fig, os.path.join(out_dir, f"deltaRho_{halo_id}.png"))

    # plano XY coloreado por deltaRho
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df_lim.x, df_lim.y, s=0.45, c=df_lim.deltaRho, cmap="viridis")
    ax.set(
        title=f"Halo {halo_id}: plano XY (Δρ)",
        xlabel="x [kpc]", ylabel="y [kpc]"
    )
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="deltaRho")
    save_fig(fig, os.path.join(out_dir, f"xy_deltaRho_{halo_id}.png"))

    log(f"=== Fin halo {halo_id}. Tiempo: {time.time() - t0:.1f}s ===\n")


def main() -> None:
    ap = argparse.ArgumentParser("Batch TNG50")
    ap.add_argument("--ids", nargs="+", required=True, help="Lista de IDs de halos")
    ap.add_argument("--in-dir", required=True, help="Directorio con CSV de entrada")
    ap.add_argument("--out-dir", default="output", help="Directorio de salida")
    args = ap.parse_args()

    for h in args.ids:
        process_halo(h, args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
