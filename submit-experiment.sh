#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

#SBATCH -A project-name # UPDATE THIS
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --mem 64G
#SBATCH --gpus h100:1

## Activate virtual environment
source /path/to/miniconda3/etc/profile.d/conda.sh # UPDATE THIS
conda activate semgaze

## Create a timestamp-based folder for the experiment
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
EXPERIMENT_PATH="experiments/$DATE/$TIME"
mkdir -p "$EXPERIMENT_PATH"

## Copy a code snapshot
cp -r "semgaze" "main.py" "submit-experiment.sh" "$EXPERIMENT_PATH"

## Switch to the experiment folder
cd "$EXPERIMENT_PATH"

## Launch experiment
python main.py --config-name "config_gf" # PDATE THIS (if needed)