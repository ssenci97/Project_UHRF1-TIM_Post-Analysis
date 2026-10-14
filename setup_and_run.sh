#!/bin/bash
#SBATCH --job-name=do
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=bbbb.log

set -euo pipefail

# 1. Create and setup venv only if it doesn't already exist
if [ ! -d "../env" ]; then
    echo "==> Creating virtual environment in ../env..."
    python3 -m venv ../env

    echo "==> Activating virtual environment..."
    source ../env/bin/activate

    echo "==> Installing dependencies..."
    pip install --upgrade pip
    pip install matplotlib numpy pyyaml biopython tqdm pymol-open-source
else
    echo "==> Virtual environment already exists. Skipping setup..."
    echo "==> Activating virtual environment..."
    source ../env/bin/activate
fi

# 2. Run the script for all directories in data/pdbs_by_group/ except 'others'
echo "==> Running view_interfaces.py processing loop..."
for d in data/pdbs_by_group/*; do 
    if [ -d "$d" ] && [ "$(basename "$d")" != "others" ]; then
        echo "Processing: $d"
        python3 src/view_interfaces.py --input "$d" --output "$d/interfaces"
    fi
done

echo "==> All tasks completed successfully!"


