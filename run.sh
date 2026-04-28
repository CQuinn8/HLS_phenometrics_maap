#!/usr/bin/env bash
# Example usage:
# bash run.sh 18SUJ 2020
#             <tile> <target_year>
set -euo pipefail

basedir=$(dirname "$(readlink -f "$0")")

if [[ $# -ne 2 ]]; then
    echo "Error: Expected 2 arguments, got $#"
    echo "Usage: $0 <tile> <target_year>"
    exit 1
fi

# default worker = 32vCPUs but I/O issues when nCPU is 31
NPROC=$(nproc)
if [[ $NPROC -le 8 ]]; then
    N_WORKERS=$(( NPROC - 1 ))
else
    N_WORKERS=$(( NPROC * 3 / 4 )) 
fi
echo "vCPUs: $NPROC  Workers: $N_WORKERS"
 
tile="$1"
target_year="$2"
chunk_size=1230

OUTPUT_DIR=output
INPUT_DIR=input
mkdir -p $OUTPUT_DIR
mkdir -p $INPUT_DIR

# 0. Set up env: in build_env.sh using uv package manager
unset PROJ_LIB
unset PROJ_DATA

# Logging
LOG_FILE="${OUTPUT_DIR}/run_${tile}_${target_year}_$(date +%Y%m%d_%H%M%S).log"
S3_LOG="${OUTPUT_DIR}/test_log.log" # "s3://maap-ops-workspace/shared/colinquinn/logs/run_${tile}_${target_year}.log"

exec > >(tee -a "$LOG_FILE") 2>&1
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    aws s3 cp "$LOG_FILE" "$S3_LOG" 2>/dev/null &
}

log "Checking credentials..."
[ -f /home/ops/.netrc ] && log "netrc: OK" || log "WARNING: .netrc not mounted"

log "===== Pipeline Started ====="
log "Tile:       $tile"
log "Year:       $target_year"
log "Input dir:  $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "Basedir:    $basedir"

# 1a. Download HLS Scenes and compute EVI2
# take target_year and generate start/end dates +/- 1 year, check for boundaries 
prev_year=$(( target_year - 1 ))
next_year=$(( target_year + 1 ))

log "Stage 1: HLS download"
cmd_download=(
    uv run --no-dev "${basedir}/getHLS.sh"
    $tile 
    "$prev_year-01-01" 
    "$next_year-12-31" 
    "${INPUT_DIR}"
)
UV_PROJECT="${basedir}" "${cmd_download[@]}"


log "Stage 2: EVI2 calculations"
cmd_calculate_evi=(
    uv run --no-dev "${basedir}/calculate_evi.py"
    --tile=$tile 
    --indir=$INPUT_DIR
    --outdir=$INPUT_DIR
    --n_workers=$(( $(nproc)/2 - 1 ))
)
UV_PROJECT="${basedir}" "${cmd_calculate_evi[@]}"


log "Stage 3: Calculating phenometrics"
cmd=(
    uv run --no-dev "${basedir}/run_phenometrics.py"
    --data_dir="${INPUT_DIR}"
    --output_path="${OUTPUT_DIR}"
    --tile="${tile}"
    --target_year="${target_year}"
    --context_months=12
    --chunk_size="${chunk_size}"
    --n_workers=20 #$N_WORKERS
)

UV_PROJECT="${basedir}" "${cmd[@]}"

log "COMPLETE"
log "===== Pipeline Complete ====="
exit 0