#!/bin/bash
#SBATCH --job-name=do
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=bbbb.log

set -euo pipefail

#python3 src/new_sync_pdbs.py
python3 src/interfaces.py --input data/models_by_group/TIM-NTD_x_UHRF1-UBL --output data/models_by_group/TIM-NTD_x_UHRF1-UBL/interfaces
python3 src/interfaces.py --input data/models_by_group/TIM-NTD_x_UHRF1-SRA --output data/models_by_group/TIM-NTD_x_UHRF1-SRA/interfaces

