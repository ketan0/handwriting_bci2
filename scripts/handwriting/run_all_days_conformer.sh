#!/bin/bash

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate bci

python -m neuralDecoder.main \
    model=conformer \
    dataset=handwriting_all_days \
    batchSize=48 \
    dataset.syntheticMixingRate=0 \
    outputDir=/Users/ketanagrawal/CS224s/run_all_days_conformer_output
