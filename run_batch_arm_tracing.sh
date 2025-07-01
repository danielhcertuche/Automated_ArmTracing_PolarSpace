#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_batch_arm_tracing.sh – Ejecuta arm_tracing_pipeline_polar_.py
#                             en paralelo dentro de sesiones *screen*,
#                             una ventana por ID.
#
# Uso:
#   ./run_batch_arm_tracing.sh 11 117251 372755 388545
#
# Después:
#   screen -r arms              # para adjuntarse y monitorear
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
shopt -s nullglob

# ─── Configuración ───────────────────────────────────────────────────────────
PYTHON="python3"                       # Intérprete Python
SCRIPT="arm_tracing_pipeline_polar_.py"  # Script principal
OUT_ROOT="./figures_batch"             # Carpeta raíz resultados
LOG_DIR="$OUT_ROOT/logs"               # Carpeta logs
SESSION="arms"                         # Nombre de la sesión screen
WORKERS=6                              # Máximo de trabajos simultáneos
# ─────────────────────────────────────────────────────────────────────────────

# 1) Comprobaciones básicas
[[ $# -ge 1 ]] || { echo "Uso: $0 <id1> [id2 ...]"; exit 1; }
command -v screen >/dev/null || { echo "ERROR: screen no está instalado"; exit 2; }
[[ -f $SCRIPT ]] || { echo "ERROR: No se encontró $SCRIPT"; exit 3; }

# 2) Directorios
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# 3) Crear / reutilizar la sesión principal
if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
  echo "▶️  Reutilizando sesión '$SESSION'"
else
  screen -dmS "$SESSION"
  echo "✅  Sesión screen '$SESSION' creada"
fi

# 4) Función que lanza un ID en una ventana separada
launch_job() {
  local id="$1"
  local job="job_${id}"
  local out_dir="$OUT_ROOT/$id"
  local log_file="$LOG_DIR/${job}.log"

  mkdir -p "$out_dir"

  # Evitar duplicados
  if screen -list | grep -q "\.${job}[[:space:]]"; then
    echo "⚠️  $job ya está en ejecución, saltando"
    return
  fi

  echo "🚀  Lanzando $job"
  screen -S "$SESSION" -dm -t "$job" bash -c "
    echo \"[START] \$(date)\" >> \"$log_file\"
    $PYTHON $SCRIPT \"${id}\" --out \"$out_dir\" --no-show >> \"$log_file\" 2>&1
    echo \"[END]   \$(date)\" >> \"$log_file\"
    echo \"✔️  ID ${id} finalizado\" >> \"$log_file\"
  "
}

# 5) Bucle principal con control de concurrencia
for id in "$@"; do
  while [[ \$(screen -ls | grep -c '\.job_') -ge $WORKERS ]]; do
    sleep 2
  done
  launch_job "$id"
  sleep 0.5
done

echo "🎉  Todos los trabajos han sido encolados en la sesión '$SESSION'"
echo "💡  Usa: screen -r $SESSION  para monitorear."
