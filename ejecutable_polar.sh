#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ─── Configuración ──────────────────────────────────────────────────────────
PYTHON="python3 -u"
SCRIPT="arm_tracing_pipeline_polar_.py"
INPUT_DIR="input_polar"
OUTPUT_ROOT="results_polar"
LOG_DIR="$OUTPUT_ROOT/logs"
LISTA="ids_4.txt"

# ─── Crear directorios base ─────────────────────────────────────────────────
mkdir -p "$INPUT_DIR" "$OUTPUT_ROOT" "$LOG_DIR"

# ─── Leer lista de IDs (quita prefijo "./" y cualquier "/" final) ────────────
if [[ ! -f "$LISTA" ]]; then
  echo "❌ No se encontró el fichero de lista de vacíos: $LISTA"
  exit 1
fi
mapfile -t IDS < <(sed -E 's#^\./##; s#/$##' "$LISTA")
if [ ${#IDS[@]} -eq 0 ]; then
  echo "❌ Lista de IDs vacía en $LISTA"
  exit 1
fi

echo "▶️  IDs a reprocesar: ${IDS[*]}"

# ─── Loop secuencial sobre cada ID ──────────────────────────────────────────
for id in "${IDS[@]}"; do
  echo "────────────────────────────────────────────────────────"
  echo "🚀 Procesando halo $id …"

  # Preparar carpetas y log
  outd="$OUTPUT_ROOT/$id"
  mkdir -p "$outd"
  logfile="$LOG_DIR/job_${id}.log"
  echo "[START] $(date +'%F %T')" > "$logfile"

  # Ir a input_dir para que el script encuentre los CSV
  pushd "$INPUT_DIR" >/dev/null

  # Ejecutar Python
  $PYTHON "../$SCRIPT" "$id" --out "../$OUTPUT_ROOT/$id" --no-show >> "../$LOG_DIR/job_${id}.log" 2>&1 \
    && echo "[END]   $(date +'%F %T') (OK)" >> "../$LOG_DIR/job_${id}.log" \
    || echo "[END]   $(date +'%F %T') (ERROR)" >> "../$LOG_DIR/job_${id}.log"

  popd >/dev/null

  echo "✅ Halo $id terminado. Revisa $LOG_DIR/job_${id}.log si necesitas."
done

echo "🎉 Todos los halos de $LISTA han sido procesados."

