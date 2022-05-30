#!/bin/bash

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
