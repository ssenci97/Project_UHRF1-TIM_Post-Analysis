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

# Loop through all subdirectories in data/pdbs_by_group/
for dir in data/pdbs_by_group/*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        
        # Skip 'others' and 'other' directories
        if [[ "$dirname" != "others" && "$dirname" != "other" ]]; then
            echo "Processing: $dirname"
            python3 src/interfaces.py --input "$dir" --output "$dir/interfaces"
        fi
    fi
done

