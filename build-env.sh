#!/usr/bin/env -S bash --login
set -euo pipefail
conda clean -afy

# 1. Install system-level dependencies using apt-get 
command -v curl >/dev/null 2>&1 || { 
    echo "Installing curl..."
    conda install --solver classic -y curl
}

# 2. Install Conda packages
echo "Installing earthaccess and awscli via Conda..."
# conda install -y -c conda-forge curl earthaccess awscli
pip install earthaccess awscli

# 3. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )

# 4. install dependencies
UV_PROJECT=$basedir uv sync --no-dev

# unset PROJ env vars
unset PROJ_LIB
unset PROJ_DATA