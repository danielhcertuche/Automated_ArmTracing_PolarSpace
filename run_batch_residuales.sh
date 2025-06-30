#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ─── Configuración ────────────────────────────────────────────────────
INPUT_DIR="./input_residuales"
OUTPUT_DIR="./results_residuales"
LOG_DIR="$OUTPUT_DIR/logs"

# Número máximo de workers concurrentes
WORKERS=8

# ─── Inicialización ──────────────────────────────────────────────────
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$LOG_DIR"
echo "▶️  Input: $INPUT_DIR"
echo "▶️  Output: $OUTPUT_DIR"
echo "▶️  Logs en: $LOG_DIR"
echo "▶️  Concurrencia máxima: $WORKERS"

# ─── Función de encolado ──────────────────────────────────────────────
launch_job() {
  local csv="$1"
  local filename="$(basename "$csv")"
  local id="${filename#halo_}"
  id="${id%%_datos_hull_withindex.csv}"
  local job_name="job_${id}"
  local job_log="$LOG_DIR/${job_name}.log"
  local job_out="$OUTPUT_DIR/$id"

  # Crear carpeta de salida para este ID
  mkdir -p "$job_out"

  # Evita duplicar sesión
  if screen -ls | grep -q "\.${job_name}[[:space:]]"; then
    echo "⚠️  Sesión $job_name ya activa, saltando."
    return
  fi

  echo "🏷  Lanzando $job_name"
  screen -dmS "$job_name" bash -c "
    echo \"[START] \$(date)\" >> \"$job_log\"
    python3 residuales_version_server.py --id \"$id\" --in-dir \"$INPUT_DIR\" --out-dir \"$job_out\" \
      >> \"$job_log\" 2>&1
    echo \"[END]   \$(date)\" >> \"$job_log\"
  "
}

# ─── Bucle principal ──────────────────────────────────────────────────
files=("$INPUT_DIR"/halo_*_datos_hull_withindex.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "❌  No hay archivos CSV en $INPUT_DIR"
  exit 1
fi

for csv in "${files[@]}"; do
  # Controla concurrencia
  while [ "$(screen -ls | grep -c '\.job_')" -ge "$WORKERS" ]; do
    sleep 2
  done

  launch_job "$csv"
  sleep 1
done

echo "✅  Todas las tareas encoladas. Usa 'screen -ls' para ver sesiones activas."
