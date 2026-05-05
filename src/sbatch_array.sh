#!/bin/bash
#SBATCH --job-name model_selection
#SBATCH --output log/log.%A_%a.out
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 6-12:00:00
#SBATCH --array=0-256%16


source env.sh
python model_selection.py