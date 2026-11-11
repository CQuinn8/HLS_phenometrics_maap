#!/usr/bin/env bash
# Example usage:
# bash run.sh 18SUJ 2020
#             <tile> <target_year>
set -euo pipefail

basedir=$(dirname "$(readlink -f "$0")")

# --- 1. Argument Parsing ---
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Error: Expected 2 - 4  arguments, got $#"
    echo "Usage: $0 <tile> <target_year> [save_scenes=false] [output_prefix=shared/colinquinn/HLS_phenometrics]"
    exit 1
fi
tile="$1"
target_year="$2"
save_scenes="${3:-false}"
output_prefix="${4:-}"

# --- 2. Configuration & Paths ---
basedir=$(dirname "$(readlink -f "$0")")
chunk_size=1220
context_months=12

OUTPUT_DIR="output"
INPUT_DIR="input"

# --- 3. Compute Resources ---
NPROC=$(nproc)
DOWNLOAD_CAP=8

# Total Workers
if [[ $NPROC -le 8 ]]; then
    N_WORKERS=$(( NPROC - 1 ))
else
    N_WORKERS=$(( NPROC * 3 / 4 )) 
fi

# Download Workers
if [[ $NPROC -le 4 ]]; then
    N_DOWNLOAD_WORKERS=1                                   
elif [[ $NPROC -le 8 ]]; then
    N_DOWNLOAD_WORKERS=$(( NPROC / 2 ))
else
    N_DOWNLOAD_WORKERS=$(( NPROC < DOWNLOAD_CAP ? NPROC : DOWNLOAD_CAP ))
fi

echo "CPUs         : $NPROC"
echo "Download     : $N_DOWNLOAD_WORKERS workers (cap=${DOWNLOAD_CAP})"
echo "Phenometrics : $N_WORKERS workers"

# --- 4. Setup Directories ---
mkdir -p "$OUTPUT_DIR" "$INPUT_DIR"
if [[ "${save_scenes,,}" == "true" || "${save_scenes}" == "1" ]]; then
    SCENES_DIR="$OUTPUT_DIR"
else
    SCENES_DIR="$INPUT_DIR"
fi


# --- 5. Logging & Environment Setup ---
unset PROJ_LIB
unset PROJ_DATA

LOG_FILE="${OUTPUT_DIR}/run_${tile}_${target_year}_$(date +%Y%m%d_%H%M%S).log"
S3_LOG="s3://maap-ops-workspace/shared/colinquinn/logs/${tile}_${target_year}.log"

exec > >(tee -a "$LOG_FILE") 2>&1
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    aws s3 cp "$LOG_FILE" "$S3_LOG" 2>/dev/null &
}

# --- 6. Pipeline Execution ---
log "===== Pipeline Started ====="
log "Tile:       $tile"
log "Year:       $target_year"
log "Save scenes: $save_scenes -> using $SCENES_DIR"
log "Input dir:  $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "Basedir:    $basedir"

prev_year=$(( target_year - 1 ))
next_year=$(( target_year + 1 ))

log "Stage 1: HLS download and Stage 2 EVI calculation"

cmd_download=(
    uv run --no-dev "${basedir}/hls_download_scenes_dps.py"
    --mgrs_tile="$tile"
    --start_date="$prev_year-01-01" 
    --end_date="$next_year-12-31" 
    --output_dir="$SCENES_DIR"
    --n_workers="$N_DOWNLOAD_WORKERS"
)
UV_PROJECT="${basedir}" "${cmd_download[@]}"
# hls_download_scenes_dps.py --mgrs_tile=18SUJ --start_date=2024-01-01 --end_date=2024-01-31 --output_dir=temp_test --n_workers=2

log "Stage 3: Calculating phenometrics"
cmd=(
    uv run --no-dev "${basedir}/run_phenometrics.py"
    --data_dir="${SCENES_DIR}"
    --output_path="${OUTPUT_DIR}"
    --tile="${tile}"
    --target_year="${target_year}"
    --context_months="${context_months}"
    --chunk_size="${chunk_size}"
    --n_workers="$N_WORKERS"
)
# run_phenometrics.py --data_dir="/home/jovyan/code/HLS_phenometrics/11SPS-multipeak/" --output_path=output --tile=11SPS --target_year=2024 --context_months=12 --chunk_size=100 --n_workers=2

UV_PROJECT="${basedir}" "${cmd[@]}"


log "Stage 4: Uploading phenometrics to S3"
cmd=(
    uv run --no-dev "${basedir}/phenometrics_output.py" 
    --tile="${tile}"
    --results_path="${OUTPUT_DIR}"
    --output_prefix="${output_prefix}"
)
# python phenometrics_output.py --tile="11SPS" --results_path="output-test" --output_prefix="shared/colinquinn/test"

UV_PROJECT="${basedir}" "${cmd[@]}"

log "===== Pipeline Complete ====="
exit 0