#!/usr/bin/env bash
set -euo pipefail

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate bci
module load system libsndfile cuda/11.5.0 cudnn/8.3.3.40

python -m neuralDecoder.main \
    model=conformer \
    lossType=rnnt \
    dataset=handwriting_all_days \
    seed=314 \
    nBatchesToTrain=1000 \
    batchesPerVal=1000 \
    learnRateStart=$1 \
    weightDecay=$2 \
    model.model_config.encoder_dropout=$3 \
    batchSize=4 \
    dataset.syntheticMixingRate=0 \
    wandb.enabled=false \
    outputDir=$SCRATCH/CS224s/run_all_days_conformer_output
