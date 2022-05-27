#!/usr/bin/env bash
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:V100S_PCIE
#SBATCH --mem=32G

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate bci
module load cuda/11.2.0 cudnn/8.1.1.33

python -m neuralDecoder.main \
    model=gru \
    dataset=handwriting_all_days \
    batchSize=48 \
    dataset.syntheticMixingRate=0 \
    outputDir=/scratch/users/agrawalk/CS224s/run_all_days_output
