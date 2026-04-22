#!/usr/bin/env bash
# Example usage:
# bash run.sh 18SUJ 2020
#             <tile> <target_year>
set -euo pipefail

basedir=$(dirname "$(readlink -f "$0")")
echo $basedir

# Parse positional arguments
if [[ $# -ne 2 ]]; then
    echo "Error: Expected 2 arguments, got $#"
    echo "Usage: $0 <tile> <target_year>"
    exit 1
fi

tile="$1"
target_year="$2"
chunk_size=100

INPUT_DIR="./input"
OUTPUT_DIR="./output"
mkdir -p $OUTPUT_DIR
mkdir -p $INPUT_DIR

# 0. Set up env: in build_env.sh using uv package manager
unset PROJ_LIB
unset PROJ_DATA

# Logging
LOG_FILE="${OUTPUT_DIR}/run_${tile}_${target_year}_$(date +%Y%m%d_%H%M%S).log"
S3_LOG="s3://maap-ops-workspace/shared/colinquinn/logs/run_${tile}_${target_year}.log"
S3_STATUS="s3://maap-ops-workspace/shared/colinquinn/logs/status_${tile}_${target_year}.txt"

exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    aws s3 cp "$LOG_FILE" "$S3_LOG" 2>/dev/null &
}

heartbeat() {
    echo "$*" > "${OUTPUT_DIR}/status.txt"
    aws s3 cp "${OUTPUT_DIR}/status.txt" "$S3_STATUS" 2>/dev/null &
}

heartbeat "STARTED: $(date)"
log "===== Pipeline Started ====="
log "Tile:       $tile"
log "Year:       $target_year"
log "Input dir:  $INPUT_DIR"
log "Output dir: $OUTPUT_DIR"
log "Basedir:    $basedir"

# 1. Download or check for HLS Scenes
log "Stage 1: HLS download (not implemented yet)"
heartbeat "STAGE_1_SKIPPED: $(date)"

# 2. Calculate or check for EVI2
log "Stage 2: EVI2 (not implemented yet)"
heartbeat "STAGE_2_SKIPPED: $(date)"

# 3. Localize test data and run phenometrics
log "Stage 3: Localizing input data"
heartbeat "LOCALIZING_DATA: $(date)"

DATA_TEST_DIR="s3://maap-ops-workspace/shared/colinquinn/hls/testing/10day-subset-SERC/"

if [[ "$DATA_TEST_DIR" == s3://* ]]; then
    S3_FILE_COUNT=$(aws s3 ls "$DATA_TEST_DIR" --recursive | wc -l)
    log "Files at S3 source: $S3_FILE_COUNT"

    if [[ "$S3_FILE_COUNT" -eq 0 ]]; then
        log "ERROR: No files found at $DATA_TEST_DIR"
        heartbeat "FAILED_NO_S3_DATA: $(date)"
        exit 1
    fi

    aws s3 sync "$DATA_TEST_DIR" "$INPUT_DIR" --no-progress 2>&1
    SYNC_EXIT=$?

    if [[ $SYNC_EXIT -ne 0 ]]; then
        log "ERROR: S3 sync failed with exit code $SYNC_EXIT"
        heartbeat "FAILED_S3_SYNC: $(date)"
        exit 1
    fi

    LOCAL_FILE_COUNT=$(find "$INPUT_DIR" -type f | wc -l)
    log "Files localized: $LOCAL_FILE_COUNT"

    if [[ "$LOCAL_FILE_COUNT" -eq 0 ]]; then
        log "ERROR: No files found after sync"
        heartbeat "FAILED_EMPTY_SYNC: $(date)"
        exit 1
    fi
fi

heartbeat "DATA_LOCALIZED: $(find $INPUT_DIR -type f | wc -l) files - $(date)"

log "Input files:"
find "$INPUT_DIR" -type f | head -20
log "Total: $(find "$INPUT_DIR" -type f | wc -l) files"

log "Stage 3: Calculating phenometrics"
heartbeat "STARTING_PHENOMETRICS: $(date)"

cmd=(
    uv run --no-dev "${basedir}/run_phenometrics.py"
    --data_dir="${INPUT_DIR}"
    --output_path="${OUTPUT_DIR}"
    --tile="${tile}"
    --target_year="${target_year}"
    --skip_download
    --skip_evi
    --context_months=12
    --chunk_size="${chunk_size}"
)

UV_PROJECT="${basedir}" "${cmd[@]}"
PYTHON_EXIT=$?

if [[ $PYTHON_EXIT -ne 0 ]]; then
    log "ERROR: Phenometrics failed with exit code $PYTHON_EXIT"
    heartbeat "FAILED_PHENOMETRICS: $(date)"
    exit 1
fi

heartbeat "COMPLETE: $(date)"
log "===== Pipeline Complete ====="