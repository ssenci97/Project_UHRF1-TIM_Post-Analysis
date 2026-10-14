#!/bin/bash
#SBATCH --job-name=interface
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/interfaces-%j.log

set -uo pipefail

###################################### configuration
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="$PROJECT_DIR/../env"
INPUT_ROOT="$PROJECT_DIR/data/pdbs_by_group"
CAPRI_TSV="$PROJECT_DIR/data/capri_ss_all.tsv"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"
SCRIPTS=("01_align_export.py" "02_haddock_analysis.py" "03_interface_analysis.py" "04_interface_summary.py")

###################################### environment
cd "$PROJECT_DIR" || exit 1
echo "[SLURM-INFO] Working directory: $PWD"
echo "[SLURM-INFO] Started: $(date '+%Y-%m-%d %H:%M:%S')"

if [ ! -d "$VENV" ]; then
    echo "[SLURM-INFO] ERROR: Virtual environment not found: $VENV"
    exit 1
fi

source "$VENV/bin/activate" || exit 1

###################################### validation
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$PROJECT_DIR/src/$script" ]; then
        echo "[SLURM-INFO] ERROR: Missing script: $PROJECT_DIR/src/$script"
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

if ! python3 -c "import Bio, matplotlib, numpy, yaml, tqdm, pymol" >/dev/null 2>&1; then
    echo "[SLURM-INFO] ERROR: Required dependencies are unavailable in $VENV"
    python3 -c "import Bio, matplotlib, numpy, yaml, tqdm, pymol"
    exit 1
fi

###################################### processing
rm -r data/pdbs_by_group/*/haddock_analysis
rm -r data/pdbs_by_group/*/interface_summary
rm -r data/pdbs_by_group/*/interfaces

START=$(date +%s)
PROCESSED_GROUPS=0
FAILED_GROUPS=0

while IFS= read -r -d '' d; do
    GROUP=$(basename "$d")
    [[ "$GROUP" == "others" || "$GROUP" == .* ]] && continue

    PROCESSED_GROUPS=$((PROCESSED_GROUPS + 1))
    echo "[SLURM-INFO] Processing group: $GROUP"
    GROUP_FAILED=0

    if python3 "src/${SCRIPTS[0]}" --input "$d" --output "$d/aligned"; then
        echo "[SLURM-INFO] $GROUP: alignment completed"
    else
        echo "[SLURM-INFO] ERROR: $GROUP: alignment failed"
        GROUP_FAILED=1
    fi

    if [ "$GROUP_FAILED" -eq 0 ]; then
        if python3 "src/${SCRIPTS[1]}" --input "$d" --output "$d/haddock_analysis" --capri-tsv "$CAPRI_TSV"; then
            echo "[SLURM-INFO] $GROUP: HADDOCK analysis completed"
        else
            echo "[SLURM-INFO] ERROR: $GROUP: HADDOCK analysis failed"
            GROUP_FAILED=1
        fi
    fi

    if [ "$GROUP_FAILED" -eq 0 ]; then
        if python3 "src/${SCRIPTS[2]}" --input "$d" --output "$d/interfaces" --workers "$WORKERS"; then
            echo "[SLURM-INFO] $GROUP: interface analysis completed"
        else
            echo "[SLURM-INFO] ERROR: $GROUP: interface analysis failed"
            GROUP_FAILED=1
        fi
    fi

    if [ "$GROUP_FAILED" -eq 0 ]; then
        if python3 "src/${SCRIPTS[3]}" --input "$d" --output "$d/interface_summary"; then
            echo "[SLURM-INFO] $GROUP: summary completed"
        else
            echo "[SLURM-INFO] ERROR: $GROUP: summary failed"
            GROUP_FAILED=1
        fi
    fi

    if [ "$GROUP_FAILED" -ne 0 ]; then
        FAILED_GROUPS=$((FAILED_GROUPS + 1))
    fi
done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

ELAPSED=$(( $(date +%s) - START ))
echo "[SLURM-INFO] Finished: $(date '+%Y-%m-%d %H:%M:%S') execution_time=${ELAPSED}s groups=$PROCESSED_GROUPS failed_groups=$FAILED_GROUPS"

if [ "$PROCESSED_GROUPS" -eq 0 ]; then
    echo "[SLURM-INFO] ERROR: No eligible group directories found"
    exit 1
fi

if [ "$FAILED_GROUPS" -gt 0 ]; then
    exit 1
fi


