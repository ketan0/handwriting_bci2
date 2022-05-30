#!/usr/bin/env bash
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:V100S_PCIE
#SBATCH --mem=32G
set -euo pipefail

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate bci
module load system libsndfile cuda/11.5.0 cudnn/8.3.3.40

python -m neuralDecoder.main \
    model=conformer \
    lossType=rnnt \
    dataset=handwriting_all_days \
    batchSize=4 \
    dataset.syntheticMixingRate=0 \
    outputDir=$SCRATCH/CS224s/run_all_days_conformer_output
