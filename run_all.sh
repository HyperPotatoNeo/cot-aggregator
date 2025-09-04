#!/bin/bash

for seed in {1234..1254}; do
    sbatch --time=12:0:0 run.sh 16 4 10 $seed
done

# for seed in 1234 1235 1236 1237 1238; do
#     for k in 1 2 3 4 5 6; do
#         sbatch --time=12:0:0 run.sh 32 $k 5 $seed
#     done

#     for n in 4 6 8 16 32; do
#         sbatch --time=12:0:0 run.sh $n 4 5 $seed
#     done

#     sbatch --time=12:0:0 run.sh 64 4 10 $seed
# done