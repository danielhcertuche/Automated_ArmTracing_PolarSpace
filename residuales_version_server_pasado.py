#!/usr/bin/env python3
# coding: utf-8
"""
Procesador batch optimizado TNG50 sin clases
- Lee CSV de un directorio de entrada usando IDs
- Guarda figuras (.png) con .savefig()
- Realiza spline cúbica, contraste y ajustes de densidad
- Manejo de segmentos con pocos datos en ajuste lineal
- Trazabilidad y registros de análisis e información relevante
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
from datetime import datetime

# Suprimir avisos innecesarios de pandas
pd.options.mode.chained_assignment = None


def log(msg: str):
    """Imprime mensaje con timestamp para trazabilidad."""
    print(f"[{datetime.utcnow().isoformat()} UTC] {msg}")


def load_data(file_path, chunks=9):
    log(f"Iniciando lectura de datos desde '{file_path}' en {chunks} trozos")
    cols_required = ['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel','Potential','U','rho','index_id']
    # Averiguar filas totales
    total = sum(1 for _ in open(file_path)) - 1
    log(f"Total de filas (aprox): {total}")
    chunk_size = max(1, total // chunks)
    log(f"Tamaño de chunk calculado: {chunk_size}")
    reader = pd.read_csv(
        file_path,
        sep=',',
        header=0,
        usecols=cols_required,
        chunksize=chunk_size
    )
    arrays = []
    for i, chunk in enumerate(reader, start=1):
        log(f"Cargando chunk {i}: {len(chunk)} filas")
        arrays.append(chunk.values)
    if not arrays:
        raise ValueError(f"El archivo {file_path} está vacío o no contiene las columnas requeridas")
    data = np.vstack(arrays)
    log(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
    return data


def save_fig(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"Figura guardada: {path}")


def process_halo(halo_id, in_dir, out_dir):
    start_time = time.time()
    log(f"=== Iniciando procesamiento del halo {halo_id} ===")
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(in_dir, f"halo_{halo_id}_datos_dbscan_fu.csv")

    # Carga de datos
    data = load_data(file_path)
    cols = ['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel','Potential','U','rho','index_id']
    df = pd.DataFrame(data, columns=cols)
    log(f"Columns: {df.columns.tolist()}")
    log(f"Primeras 5 densidades rho: {df['rho'].head().tolist()}")

    # Cálculo de coordenadas
    df['Rs'] = np.sqrt(df.x**2 + df.y**2)
    df['Zs'] = df.z

    # Filtrado inicial
    df_sorted = df.sort_values('Rs')
    df_filtered = df_sorted[(df_sorted.Zs.abs() <= 1.5) & (df_sorted.Rs > 1)]
    log(f"Filas tras filtrado espacial: {len(df_filtered)} / {len(df_sorted)}")

    # Bin promedio de densidades
    deltaR = 0.1
    Rs_mean, rho_mean = [], []
    step = 0.0
    while step + deltaR < 20:
        mask = (df_filtered.Rs > step) & (df_filtered.Rs < step + deltaR)
        if mask.any():
            center = round((df_filtered.Rs[mask].min() + df_filtered.Rs[mask].max()) * 0.5, 2)
            mean_rho = df_filtered.rho[mask].mean()
            Rs_mean.append(center)
            rho_mean.append(mean_rho)
        step += deltaR
    Rs_mean = np.array(Rs_mean)
    rho_mean = np.array(rho_mean)
    log(f"Puntos promedio calculados: {len(Rs_mean)}")
    ylog = np.log10(rho_mean)

    # ===== Spline cúbica y métricas =====
    log("Construyendo spline cúbica")
    spline = interpolate.interp1d(Rs_mean, rho_mean, kind='cubic', fill_value='extrapolate')
    # evaluar spline en los mismos Rs_mean para métricas
    rho_fit = spline(Rs_mean)
    # r2_spline
    ss_res = np.sum((rho_mean - rho_fit)**2)
    ss_tot = np.sum((rho_mean - rho_mean.mean())**2)
    r2_spline = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    # chi2_spline
    chi2_spline = ss_res
    log(f"Métrica spline: r2_spline={r2_spline:.4f}, chi2_spline={chi2_spline:.4e}")

    xnew = np.linspace(Rs_mean.min(), Rs_mean.max(), 2000)
    ynew = spline(xnew)

    # Gráfica ρ vs R
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(df_sorted.Rs.values, df_sorted.rho.values, s=0.3, alpha=0.45, label='Data')
    ax.scatter(Rs_mean, rho_mean, facecolors='none', edgecolors='k', marker='v', label='Mean ρ')
    ax.plot(xnew, ynew, linewidth=1.5, alpha=0.8, label='Spline')
    ax.set(title=f'Halo {halo_id}: ρ vs R', xlabel='R [kpc]', ylabel='ρ')
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, os.path.join(out_dir, f'rho_vs_R_{halo_id}.png'))

    # Contraste y ajustes
    df_lim = df_filtered[(df_filtered.Rs > Rs_mean.min()) & (df_filtered.Rs < Rs_mean.max())].copy()
    df_lim['rho_sph_log'] = np.log10(df_lim.rho)
    df_lim['deltaRho'] = df_lim.rho / spline(df_lim.Rs) - 1.0
    log(f"Valores de deltaRho calculados: min={df_lim.deltaRho.min():.3f}, max={df_lim.deltaRho.max():.3f}")

    # Filtrado de contraste
    filt = df_lim[(df_lim.deltaRho > 0) & (df_lim.deltaRho < 20)].copy()
    bg = df_lim[df_lim.deltaRho <= 0].copy()

    # Ajuste lineal por tramos
    cuts = [3, 4, 7.75, Rs_mean.max()]
    def line(x, a, b): return a*x + b
    params = []
    log("Iniciando ajuste lineal por tramos:")
    for i in range(len(cuts)):
        low = cuts[i-1] if i>0 else -np.inf
        mask = (Rs_mean > low) & (Rs_mean <= cuts[i])
        x_seg, y_seg = Rs_mean[mask], ylog[mask]
        if x_seg.size >= 2:
            popt, _ = optimize.curve_fit(line, x_seg, y_seg)
            log(f"  Segmento {i+1}: rango ({low},{cuts[i]}), parámetros a={popt[0]:.4f}, b={popt[1]:.4f}")
        else:
            popt = np.array([0.0, 0.0])
            log(f"  Segmento {i+1}: datos insuficientes, usando fallback a=0, b=0")
        params.append(popt)

    # Aplicar residual
    a_vals = [p[0] for p in params]
    b_vals = [p[1] for p in params]
    for idx, (a, b) in enumerate(zip(a_vals, b_vals), start=1):
        col = f'rho_resta{idx}'
        filt[col] = filt.rho_sph_log - (a*filt.Rs + b) + filt.rho_sph_log.min()

    # Selección de residual final
    conds = [
        filt.Rs < cuts[0],
        (filt.Rs >= cuts[0]) & (filt.Rs < cuts[1]),
        (filt.Rs >= cuts[1]) & (filt.Rs <= cuts[2]),
        (filt.Rs >= cuts[2]) & (filt.Rs <= cuts[3]),
    ]
    choices = [f'rho_resta{i}' for i in range(1,5)]
    filt['rho_resta_final'] = np.select(conds, [filt[c] for c in choices])
    filt['rho_resta_final_exp'] = 10**filt.rho_resta_final

    # Guardar resultados
    csv1 = os.path.join(out_dir, f'data_rho_{halo_id}_.csv')
    filt[['x','y','z','Rs','vx','vy','vz','rho_resta_final_exp']].to_csv(csv1, index=False)
    log(f"Guardado CSV resumido: {csv1} ({len(filt)} filas)")

    csv2 = os.path.join(out_dir, f'data_rho_{halo_id}_filtered.csv')
    cols2 = ['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel',
             'Potential','U','rho','Rs','Zs','rho_sph_log','deltaRho'] + choices + ['rho_resta_final','rho_resta_final_exp','index_id']
    filt[cols2].to_csv(csv2, index=False)
    log(f"Guardado CSV completo: {csv2} ({len(filt)} filas)")

    # Gráfica deltaRho
    fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(df_lim.Rs, df_lim.deltaRho, s=0.45, c=df_lim.deltaRho, cmap='plasma')
    ax.set(title=f'Halo {halo_id}: DeltaRho contraste', xlabel='R [kpc]', ylabel='deltaRho')
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label='deltaRho')
    save_fig(fig, os.path.join(out_dir, f'deltaRho_{halo_id}.png'))

    elapsed = time.time() - start_time
    log(f"=== Fin procesamiento halo {halo_id}. Tiempo: {elapsed:.1f}s ===\n")


def parse_args():
    p = argparse.ArgumentParser(description='Batch TNG50 por ID y directorio')
    p.add_argument('--ids', nargs='+', required=True, help='Lista de IDs de halos')
    p.add_argument('--in-dir', required=True, help='Directorio con archivos CSV')
    p.add_argument('--out-dir', default='output', help='Directorio de salida')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for hid in args.ids:
        process_halo(hid, args.in_dir, args.out_dir)
