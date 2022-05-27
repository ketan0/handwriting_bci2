#!/bin/bash

python -m neuralDecoder.main \
    model=conformer \
    lossType=rnnt \
    dataset=handwriting_all_days \
    batchSize=2 \
    dataset.syntheticMixingRate=0 \
    outputDir=/Users/ketanagrawal/CS224s/run_all_days_conformer_output
