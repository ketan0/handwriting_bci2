#!/usr/bin/env bash
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:V100S_PCIE
#SBATCH --mem=32G

conda activate base
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate bci
module load system sndfile cuda/11.2.0 cudnn/8.1.1.33

python -m neuralDecoder.main \
    model=gru \
    dataset=handwriting_all_days \
    batchSize=48 \
    dataset.syntheticMixingRate=0 \
    outputDir=$SCRATCH/CS224s/run_all_days_output
