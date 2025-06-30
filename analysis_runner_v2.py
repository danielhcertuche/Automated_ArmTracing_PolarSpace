#!/usr/bin/env python3
"""
Automatización del análisis de subhalos de Illustris-TNG50.
Soporta un archivo (.gas) o un directorio completo, validando antes y mostrando prints de progreso.
La lógica científica de las funciones permanece inalterada.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import logging
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ROUTINES.data_preprocessing.tng50_data_processor import TNG50DataProcessor
from ROUTINES.data_preprocessing.rotation_handler import RotationHandler
from ROUTINES.data_preprocessing.velocity_escape_calculator import VelocityEscapeCalculator
from ROUTINES.filtering.filtering import (
    create_velocity_grid,
    filter_data_by_velocity,
    filter_data_by_velocity_index,
    apply_convex_hull_filter
)
from ROUTINES.visualization.visualization_handler import visualize_convex_hull
from ROUTINES.filtering.density_based_spatial_clustering import (
    apply_dbscan_filter,
    Points_no_outliers
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analysis runner: archivo único o carpeta de .gas'
    )
    parser.add_argument('path',
                        help='Ruta a un .gas o a un directorio con archivos .gas')
    parser.add_argument('--eps', type=float, required=True,
                        help='EPS para DBSCAN')
    parser.add_argument('--min', dest='min_samples', type=int, required=True,
                        help='min_samples para DBSCAN')
    parser.add_argument('-o', '--out', required=True,
                        help='Directorio de salida para CSVs e imágenes')
    return parser.parse_args()


def collect_files(path):
    if os.path.isfile(path):
        if not path.endswith('.gas'):
            raise ValueError(f'El archivo proporcionado no es .gas: {path}')
        return [path]
    if os.path.isdir(path):
        pattern = os.path.join(path, '*.gas')
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f'No se encontraron archivos .gas en {path}')
        return files
    raise FileNotFoundError(f'Ruta no existe: {path}')


def setup_logging(out_dir: str, halo_id: str):
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'subhalo_{halo_id}.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.info(f'=== Inicio Subhalo {halo_id} ===')


def plot_and_save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f'Figura guardada: {path}')

    
def plot_orthogonal_projections(df, halo_id: str, prefix: str, out_dir: str):
     
    max_x = np.max(np.abs(df['x'].values))
    if max_x > 50:
        limit = 40
    else:
        limit = max_x + 5.0
    limits = [-limit, limit]
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    axes = [('x', 'y'), ('x', 'z'), ('y', 'z')]
    titles = ['Orthogonal Projection: XY', 'Orthogonal Projection: XZ', 'Orthogonal Projection: YZ']

    for ax, color, (ax_x, ax_y), title in zip(axs, colors, axes, titles):
        ax.scatter(df[ax_x], df[ax_y], s=0.05, color=color, alpha=0.6)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(f'{ax_x.upper()} [kpc]', fontsize=14)
        ax.set_ylabel(f'{ax_y.upper()} [kpc]', fontsize=14)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in', length=6)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.2)

    filename = f'{prefix.lower()}_{halo_id}_ortho.png'
    plot_and_save(fig, out_dir, filename)



def process_velocity_pipeline(df: pd.DataFrame, halo_id: str, out_dir: str, file_path: str):
    print(f'[{halo_id}] Iniciando cálculos de velocidades')
    logging.info('Iniciando calculos:')
    # Mantener exactamente la lógica interna
    simulation = TNG50DataProcessor(data = df)
    pos_rot = simulation.accumulated_data[:, :3]
    data_rot = simulation.accumulated_data[:, 0:12]
    index_array = simulation.accumulated_data[:, 12].astype(int)
    potential = simulation.accumulated_data[:, 9]

    calc = VelocityEscapeCalculator(data_rot, potential)
    calc.calculate_magnitudes()
    calc.calculate_escape_velocity()
    processed = calc.get_processed_data()
    print(f'[{halo_id}] Cálculo de velocidades completado con exito')
  
    grid, v_esc_mid, w_avg_vel, max_avg_vel, v_circ_min, v_circ_max, v_esc_min, v_esc_max = create_velocity_grid(processed)
    filtered = filter_data_by_velocity(processed, max_avg_vel, v_esc_mid, w_avg_vel)
    filtered_index = filter_data_by_velocity_index(processed, max_avg_vel, v_esc_mid, w_avg_vel, index_array)
    hull, pts, inside, max_dist, sample_pts, df_hull_data = apply_convex_hull_filter(pos_rot, filtered, data_rot, index_array, halo_id)
    print(f'[{halo_id}] Velocidades: max_avg_vel:{max_avg_vel:.2f}, v_esc_max:{v_esc_max:.2f}')
  
    fig, ax = plt.subplots(figsize=(8, 6))
    # Mapa de calor con paleta accesible para daltónicos
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        extent=[v_esc_min, v_esc_max, v_circ_min, v_circ_max],
        cmap='cividis'
    )
    ax.scatter(
        processed['escape_velocity'],
        processed['velocity_magnitude'],
        c='crimson', s=1, alpha=0.3, label='Datos individuales'
    )
    # Línea punteada del promedio máximo
    ax.axhline(
        y=max_avg_vel,
        linestyle='--',
        color='#008080',
        linewidth=1.2,
        label=f'Máx. velocidad promedio = {max_avg_vel:.2f} km/s'
    )
    # Promedio por columna como puntos negros marcados
    ax.scatter(
        v_esc_mid,
        w_avg_vel,
        c='black',
        s=40,
        edgecolors='white',
        linewidths=0.5,
        label='Velocidad promedio por columna'
    )
    ax.set_title(f'Perfil de velocidades: {os.path.basename(file_path)}', fontsize=15)
    ax.set_xlabel(r'$V_{esc}$ [km/s]', fontsize=13)
    ax.set_ylabel(r'$V_{circ}$ [km/s]', fontsize=13)
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Densidad / Conteo', fontsize=12)

    plot_and_save(fig, out_dir, f'grid_{halo_id}.png')

    visualize_convex_hull(hull, sample_pts, pts, inside, halo_id)
    print(f'[{halo_id}] Completado filtro por velocidad')
    print('--------------------------------------------')
    
    print(f'[{halo_id}] Iniciando escritura de csv filtro de velocidad')
    csv_path = os.path.join(out_dir, f'halo_{halo_id}_datos_dbscan_vel.csv')
    df_hull_data.to_csv(csv_path, index=False)
    
    return df_hull_data


def process_dbscan_pipeline(df: pd.DataFrame, halo_id: str, eps: float, min_samples: int, out_dir: str):
    print(f'[{halo_id}] Iniciando pipeline DBSCAN sin ningun filtro(eps={eps}, min_samples={min_samples})')
    df_filtered, background, eps_val, min_val, labels = apply_dbscan_filter(df, eps=eps, min_samples=min_samples)

    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.scatter(df['x'], df['y'], s=0.01, c='blue')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('Original'); plt.xlim(-30, 30); plt.ylim(-30, 30)
    plt.subplot(1, 3, 2)
    plt.scatter(df_filtered['x'], df_filtered['y'], s=0.01, c='green')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('DBSCAN Filtered')
    plt.subplot(1, 3, 3)
    plt.scatter(background['x'], background['y'], s=0.01, c='red')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('Outliers')
    plot_and_save(fig, out_dir, f'dbscan_{halo_id}.png')
    plot_orthogonal_projections(df_filtered[['x','y','z']], halo_id, 'dbscan_filtered', out_dir)

    sim = TNG50DataProcessor(data=df_filtered)
    #Calcula centro de masa, momento angular y rota
    rotated_pos, rotated_total = sim.visualize_rotated_vector()
    rot_df = pd.DataFrame(rotated_pos, columns=['x','y','z'])
    plot_orthogonal_projections(rot_df, halo_id, 'rotated_dbscan', out_dir)

    cols = ['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel','Potential','U','rho', 'index_id']
    df_db = pd.DataFrame(rotated_total, columns=cols)
    df_db['index_id_med'] = df_db.index
    print(f'[{halo_id}] Iniciando escritura de csv')
    csv_path = os.path.join(out_dir, f'halo_{halo_id}_datos_dbscan_fu.csv')
    df_db.to_csv(csv_path, index=False)
    logging.info(f'Datos DBSCAN sin ningun filtro guardados en {csv_path}')
    
    fig_rho, ax_rho = plt.subplots(figsize=(8, 6))
    Rs = np.sqrt(df_db['x']**2 + df_db['y']**2)
    ax_rho.scatter(Rs, df_db['rho'], color='#1f77b4', s=0.05, alpha=0.43, label='Data')
    ax_rho.set_xlabel(r'$R$ [kpc]', fontsize=14)
    ax_rho.set_ylabel(r'$\rho$ [$M_\odot$ kpc$^{-3}$]', fontsize=14)
    ax_rho.set_title('Radial Density Profile (AVF)', fontsize=15)
    ax_rho.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    ax_rho.tick_params(axis='both', labelsize=12, direction='in', length=6)
    plot_and_save(fig_rho, out_dir, f'rho_{halo_id}.png')
    
    
    print(f'[{halo_id}] Completado pipeline DBSCAN 01')
    print('--------------------------------------------')
    return df_db

def process_dbscan_pipeline_after_velocity_filter(df: pd.DataFrame, halo_id: str, eps: float, min_samples: int, out_dir: str):
    print(f'[{halo_id}] Iniciando pipeline DBSCAN after_velocity_filter (eps={eps}, min_samples={min_samples})')
    df_filtered, background, eps_val, min_val, labels = apply_dbscan_filter(df, eps=eps, min_samples=min_samples)

    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.scatter(df['x'], df['y'], s=0.01, c='blue')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('Original(AVF)'); plt.xlim(-30, 30); plt.ylim(-30, 30)
    plt.subplot(1, 3, 2)
    plt.scatter(df_filtered['x'], df_filtered['y'], s=0.01, c='green')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('DBSCAN Filtered(AVF)')
    plt.subplot(1, 3, 3)
    plt.scatter(background['x'], background['y'], s=0.01, c='red')
    plt.xlabel('X [kpc]'); plt.ylabel('Y [kpc]')
    plt.title('Outliers(AVF)')
    plot_and_save(fig, out_dir, f'dbscan_{halo_id}_(AVF).png')
    plot_orthogonal_projections(df_filtered[['x','y','z']], halo_id, 'dbscan_filtered(AVF)', out_dir)    
    phys_cols = ['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel','Potential','U','rho']
    
    df_db = (df_filtered.reset_index().rename(columns={'index':'index_id_final'}).loc[:, phys_cols + ['index_id','index_idx','index_id_final']])	    
    print(f'[{halo_id}] Iniciando escritura de csv')
    csv_path = os.path.join(out_dir, f'halo_{halo_id}_datos_vel_dbscan.csv')
    df_db.to_csv(csv_path, index=False)
    logging.info(f'Datos DBSCAN guardados en {csv_path}')
    
    fig_rho, ax_rho = plt.subplots(figsize=(8, 6))
    Rs = np.sqrt(df_db['x']**2 + df_db['y']**2)
    ax_rho.scatter(Rs, df_db['rho'], color='#1f77b4', s=0.05, alpha=0.43, label='Data')
    ax_rho.set_xlabel(r'$R$ [kpc]', fontsize=14)
    ax_rho.set_ylabel(r'$\rho$ [$M_\odot$ kpc$^{-3}$]', fontsize=14)
    ax_rho.set_title('Radial Density Profile', fontsize=15)
    ax_rho.grid(True, linestyle='--', linewidth=0.4, alpha=0.3)
    ax_rho.tick_params(axis='both', labelsize=12, direction='in', length=6)
    plot_and_save(fig_rho, out_dir, f'rho_{halo_id}.png')
    
    
    print(f'[{halo_id}] Completado pipeline DBSCAN 02')

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    try:
        files = collect_files(args.path)
    except Exception as e:
        print(f'❌ {e}', file=sys.stderr)
        sys.exit(1)

    for file_path in files:
        base = os.path.basename(file_path).replace('.gas','')
        halo_id = base.split('_')[-1]
        setup_logging(args.out, halo_id)

        print(f'[{halo_id}] Comenzando análisis de {file_path}')
        logging.info(f'Starting analysis for {file_path}')

        df = pd.read_csv(file_path, sep=' ', header=None,
                         names=['x','y','z','vx','vy','vz','lxvel','lyvel','lzvel','Potential','U','rho'])
        df_db = process_dbscan_pipeline(df, halo_id, args.eps, args.min_samples, args.out)
        
        df_hull_data = process_velocity_pipeline(df_db, halo_id, args.out, file_path)
        process_dbscan_pipeline_after_velocity_filter(df_hull_data, halo_id, 0.1, 3, args.out)
        logging.info(f'=== Fin Subhalo {halo_id} ===')
        print(f'[{halo_id}] Análisis completo')

if __name__ == '__main__':
    main()
