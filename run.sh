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
K=$3
N=$4
T=$5
seed=$6
tokens=$7

export TOKENIZERS_PARALLELISM=false
python scripts/eval_code.py --k $K --population $N --loops $T --dataset $data --model $model --seed $seed --resume --output ./evaluation/$data --max-new-tokens $tokens --self_verify