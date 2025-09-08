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

N=$1
K=$2
T=$3
seed=$4

export TOKENIZERS_PARALLELISM=false
python scripts/eval_code.py --k $K --population $N --loops $T --dataset ./data/he/test.parquet --output ./data/he/evaluation --seed $seed
python scripts/eval_code.py --k $K --population $N --loops $T --dataset ./data/mbpp/test.parquet --output ./data/mbpp/evaluation --seed $seed