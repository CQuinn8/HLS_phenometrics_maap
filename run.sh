#!/usr/bin/env
# Example usage:
# bash run.sh 18SUJ 2020 /projects/my-public-bucket/hls/testing/10day-subset-SERC/ /projects/my-public-bucket/hls/testing/10day-subset-SERC/ 
#             <tile> <target_year> <DATA_DIR> <OUTPUT_DIR>
set -euo pipefail

basedir=$(dirname "$(readlink -f "$0")")
echo $basedir

# Parse positional arguments (2 required, 2 optional)
if [[ $# -lt 2 ]] || [[ $# -gt 3 ]]; then
    echo "Error: Expected 2 arguments, got $#"
    echo "Usage: $0 <tile> <target_year>"
    exit 1
fi
 
tile="$1"
target_year="$2"
chunk_size=100

INPUT_DIR="./input"
OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

# 0. Set up env: in build_env.sh using uv package manager
unset PROJ_LIB
unset PROJ_DATA

LOG_FILE="${OUTPUT_DIR}/run_${tile}_${target_year}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Pipeline Started ====="
echo "Tile:       $tile"
echo "Year:       $target_year"
echo "Data dir:   $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Basedir:    $basedir"

# 1. Download or check for HLS Scenes
echo "Stage 1: HLS download (not implemented yet)"

# 2. Calculate or check for EVI2 
echo "Stage 2: EVI2 (not implemented yet)"

# 3. Run phenometrics
echo "Stage 3: Calcualte phenometrics"
# conda run --live-stream --name python python "${basedir}/run_phenometrics.py" --data_dir="/projects/my-public-bucket/hls/testing/10day-subset-SERC/" --output_path="/projects/my-public-bucket/HLS_phenometrics/10day_subset_SERC/" --tile=18SUJ --target_year=2020 --skip_download --skip_evi --context_months=12 --chunk_size=$chunk_size
# conda run --live-stream --name python python "${basedir}/run_phenometrics.py" --data_dir="/projects/my-public-bucket/hls/testing/10day-subset-SERC/" --output_path="output" --tile=$tile --target_year=$target_year --skip_download --skip_evi --context_months=12 --chunk_size=$chunk_size
DATA_TEST_DIR="s3://maap-ops-workspace/shared/colinquinn/hls/testing/10day-subset-SERC/"
mkdir -p "$INPUT_DIR"

if [[ "$DATA_TEST_DIR" == s3://* ]]; then
    echo "Checking S3 source..."
    S3_FILE_COUNT=$(aws s3 ls "$DATA_TEST_DIR" --recursive | wc -l)
    echo "Files at S3 source: $S3_FILE_COUNT"
    
    if [[ "$S3_FILE_COUNT" -eq 0 ]]; then
        echo "ERROR: No files found at $DATA_TEST_DIR"
        exit 1
    fi
    
    echo "Starting S3 sync..."
    aws s3 sync "$DATA_TEST_DIR" "$INPUT_DIR" --no-progress 2>&1
    
    LOCAL_FILE_COUNT=$(find "$INPUT_DIR" -type f | wc -l)
    echo "Files localized: $LOCAL_FILE_COUNT"
    
    if [[ "$LOCAL_FILE_COUNT" -eq 0 ]]; then
        echo "ERROR: No files after sync"
        exit 1
    fi
fi

echo "Input files:"
find "$DATA_DIR" -type f | head -20
echo "Total: $(find "$DATA_DIR" -type f | wc -l) files"

cmd=(
    uv run --no-dev "${basedir}/run_phenometrics.py"
    --data_dir="${INPUT_DIR}"
    --output_path="${OUTPUT_DIR}"
    --target_year="${target_year}"
    --skip_download 
    --skip_evi 
    --context_months=12 
    --chunk_size="${chunk_size}"
)

# Execute the command with UV_PROJECT environment variable
UV_PROJECT="${basedir}" "${cmd[@]}"
echo "===== Pipeline Complete ====="