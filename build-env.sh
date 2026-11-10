#!/usr/bin/env -S bash --login
set -euo pipefail

# 1. Install system-level dependencies using apt-get 
command -v curl >/dev/null 2>&1 || { 
    echo "Installing curl..."
    conda install -y --solver classic -c conda-forge libxml2 curl
}

# 2. Install Conda packages
echo "Installing earthaccess and awscli via Conda..."
conda install -y -c conda-forge curl earthaccess awscli

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env || source $HOME/.local/bin/env

# 4. Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )

# 5. Install Python dependencies
UV_PROJECT=$basedir uv sync --no-dev

# 6. Unset PROJ env vars (Prevents projection conflicts between GDAL versions)
unset PROJ_LIB
unset PROJ_DATA