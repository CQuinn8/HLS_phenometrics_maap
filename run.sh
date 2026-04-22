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

INPUT_DIR=input
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