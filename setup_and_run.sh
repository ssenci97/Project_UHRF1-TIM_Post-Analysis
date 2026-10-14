#!/bin/bash
#SBATCH --job-name=interface
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/interfaces-%j.log

set -euo pipefail

###################################### configuration
VENV="../env"
INPUT_ROOT="data/pdbs_by_group"
CAPRI_TSV="data/capri_ss_all.tsv"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"
SCRIPTS=("01_align_export.py" "02_haddock_analysis.py" "03_interface_analysis.py" "04_interface_summary.py")

###################################### environment
mkdir -p logs
if [ ! -d "$VENV" ]; then
    echo "[SLURM-INFO] Creating virtual environment: $VENV"
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
if ! python3 -c "import Bio, matplotlib, numpy, yaml, tqdm, pymol" >/dev/null 2>&1; then
    echo "[SLURM-INFO] Installing dependencies"
    python3 -m pip install --upgrade pip
    python3 -m pip install matplotlib numpy pyyaml biopython tqdm pymol-open-source
fi

###################################### validation
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "src/$script" ]; then
        echo "[SLURM-INFO] ERROR: Missing script src/$script"
        exit 1
    fi
done
if [ ! -d "$INPUT_ROOT" ]; then
    echo "[SLURM-INFO] ERROR: Input directory not found: $INPUT_ROOT"
    exit 1
fi
if [ ! -f "$CAPRI_TSV" ]; then
    echo "[SLURM-INFO] ERROR: CAPRI file not found: $CAPRI_TSV"
    exit 1
fi

###################################### processing
START=$(date +%s)
echo "[SLURM-INFO] Starting four-stage pipeline: $(date '+%Y-%m-%d %H:%M:%S')"
for d in "$INPUT_ROOT"/*; do
    [ -d "$d" ] || continue
    [ "$(basename "$d")" = "others" ] && continue
    echo "[SLURM-INFO] Processing group: $d"
    python3 src/01_align_export.py --input "$d" --output "$d/aligned" --capri-tsv "$CAPRI_TSV" --workers "$WORKERS"
    python3 src/02_haddock_analysis.py --input "$d" --output "$d/haddock_analysis" --capri-tsv "$CAPRI_TSV"
    python3 src/03_interface_analysis.py --input "$d" --output "$d/interfaces" --workers "$WORKERS"
    python3 src/04_interface_summary.py --input "$d" --output "$d/interface_summary"
done
ELAPSED=$(( $(date +%s) - START ))
echo "[SLURM-INFO] Finished: $(date '+%Y-%m-%d %H:%M:%S') execution_time=${ELAPSED}s"


