#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:4

source ~/.zshrc
module load python/3.12.4
module load cuda/12.6
module load httpproxy/1.0
module load arrow/18.1.0
module load gcc
module load opencv/4.12.0
module load rust
conda activate verl

model=$1
data=$2
seed=$3
tokens=$4

export TOKENIZERS_PARALLELISM=false
python scripts/eval_loop_cl.py --model $model --k 4 --population 16 --loops 10 --dataset ./data/$data/train.parquet --output ./context-length_evaluation/$data --seed $seed --max-new-tokens $tokens