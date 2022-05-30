#!/bin/bash
set -euo pipefail

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate bci
module load system libsndfile cuda/11.2.0 cudnn/8.1.1.33

python -m neuralDecoder.main \
    model=conformer_toy \
    overfitBatch=True \
    lossType=rnnt \
    dataset=handwriting_single_day \
    dataset.subsetSize=1 \
    seed=1 \
    batchSize=1 \
    dataset.syntheticMixingRate=0 \
    outputDir=/Users/ketanagrawal/CS224s/run_all_days_conformer_output
