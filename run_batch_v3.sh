#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ----------------------------------------
# Configuración centralizada
# ----------------------------------------
INPUT_PATH="./input"        # Directorio o archivo .gas único
OUT_DIR="./output"          # Directorio base de salida
EPS=0.17                    # Parámetro EPS para DBSCAN
MIN_SAMPLES=20              # Parámetro min_samples para DBSCAN
WORKERS=4                   # Máximo de procesos en paralelo
SKIP_EXISTING=false         # Saltar halos ya procesados
# ----------------------------------------

usage() {
  cat <<EOF
Uso: $0 [--input <ruta>] [--out <dir>] [--eps <float>] [--min <int>] [--workers <int>] [--skip-existing]
EOF
  exit 1
}

# Parseo de argumentos
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)         INPUT_PATH="$2";    shift 2;;
    --out)           OUT_DIR="$2";       shift 2;;
    --eps)           EPS="$2";           shift 2;;
    --min)           MIN_SAMPLES="$2";   shift 2;;
    --workers)       WORKERS="$2";       shift 2;;
    --skip-existing) SKIP_EXISTING=true; shift   ;;
    *)               usage;;
  esac
done

echo "▶️ Input:         $INPUT_PATH"
echo "▶️ Output:        $OUT_DIR"
echo "▶️ DBSCAN eps:    $EPS"
echo "▶️ DBSCAN min:    $MIN_SAMPLES"
echo "▶️ Concurrencia:  $WORKERS procesos"
echo "▶️ Skip existing: $SKIP_EXISTING"
echo

# Construir lista de archivos .gas
if [[ -d "$INPUT_PATH" ]]; then
  FILES=( "$INPUT_PATH"/*.gas )
elif [[ -f "$INPUT_PATH" && "$INPUT_PATH" == *.gas ]]; then
  FILES=( "$INPUT_PATH" )
else
  echo "❌ '$INPUT_PATH' no es un directorio ni un .gas válido"
  exit 1
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "❌ No se encontraron .gas en '$INPUT_PATH'"
  exit 1
fi

# Función para encolar un subhalo
enqueue() {
  local file="$1"
  local base=$(basename "$file" .gas)
  local halo_id="${base##*_}"
  local eps_local=$EPS
  local min_local=$MIN_SAMPLES

  # Si el ID está en la lista, ajusta eps y min_samples
  if grep -qx "$halo_id" ids_logmass0_gt11.txt; then
    eps_local=0.17
    min_local=10
  fi

  local halo_out="$OUT_DIR/$halo_id"
  mkdir -p "$halo_out"/results "$halo_out"/logs

  if $SKIP_EXISTING && [[ -f "$halo_out/logs/results_done.flag" ]]; then
    echo "⚠️  Halo $halo_id ya procesado, saltando."
    return
  fi

  echo "🏷 Encolando Subhalo $halo_id"
  screen -dmS "sub_${halo_id}" bash -lc "\
echo \"[${halo_id}] Inicio...\" && \
python3 analysis_runner_v2.py \
  --eps $eps_local \
  --min $min_local \
  -o '$halo_out/results' \
  '$file' \
>> '$halo_out/logs/${halo_id}.log' 2>&1 && \
touch '$halo_out/logs/results_done.flag'"
}

# Encolar con límite de concurrencia
for f in "${FILES[@]}"; do
  while [ "$(screen -ls | grep -c 'sub_')" -ge "$WORKERS" ]; do
    sleep 2
  done
  enqueue "$f"
  sleep 0.5
done

echo
echo "✅ Todos los subhalos encolados (máx. $WORKERS en paralelo)."
echo "➡️ Monitor: screen -ls"

