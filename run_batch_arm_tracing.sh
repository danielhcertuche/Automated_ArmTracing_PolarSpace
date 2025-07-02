#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ─── Configuración ─────────────────────────────────────────────────────────
PYTHON="python3"
SCRIPT="./arm_tracing_pipeline_polar_.py"
INPUT_DIR="./input_polar"
OUTPUT_DIR="./results_polar"
LOG_DIR="$OUTPUT_DIR/logs"
SESSION="halo_batch"
WORKERS=8   # Máximo de sesiones screen simultáneas
# ─────────────────────────────────────────────────────────────────────────────

# Crear carpetas si no existen
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# Extraer lista de IDs
IDS=( $(ls "$INPUT_DIR"/data_rho_*_filtered.csv 2>/dev/null \
         | sed -E 's@.*/data_rho_([0-9]+)_filtered.csv@\1@') )
if [ ${#IDS[@]} -eq 0 ]; then
  echo "❌ No se encontraron archivos *_filtered.csv en $INPUT_DIR"
  exit 1
fi

echo "▶️  IDs a procesar: ${IDS[*]}"
echo "▶️  Sesión screen: $SESSION (máx. $WORKERS ventanas)"

# Crear o reutilizar sesión 'maestra'
if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
  echo "♻️  Reutilizando sesión screen \"$SESSION\""
else
  screen -dmS "$SESSION"
  echo "✅  Creada sesión screen \"$SESSION\""
fi

# Conteo de ventanas 'halo_' activas
count_halos(){
  screen -ls | grep -c 'halo_[0-9]\+'
}

# Lanzar cada halo en su propia ventana
for id in "${IDS[@]}"; do
  # Esperar si ya hay >= WORKERS ventanas activas
  while [ "$(count_halos)" -ge "$WORKERS" ]; do
    sleep 1
  done

  win="halo_${id}"
  outd="$OUTPUT_DIR/$id"
  logf="$LOG_DIR/job_${id}.log"
  mkdir -p "$outd"

  echo "🚀 Encolando $win"

  screen -S "$SESSION" -X screen -t "$win" bash -lc "
    echo '[START] ' \$(date +'%F %T') > '$logf'
    $PYTHON -u $SCRIPT '$id' --out '$outd' --no-show >> '$logf' 2>&1
    ec=\$?
    echo '[END]   ' \$(date +'%F %T') \"(exit \$ec)\" >> '$logf'
    exit \$ec
  "
done

echo "🎉 Todos los trabajos han sido encolados en session \"$SESSION\"."
echo "   Para verlas: screen -r $SESSION"

