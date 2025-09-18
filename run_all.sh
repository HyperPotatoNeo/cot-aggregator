#!/bin/bash

# for seed in 1234 1235 1236 1237; do
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 1 160 1 $seed 8192
    # sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-30B-A3B-Instruct-2507 lcb 1 160 1 $seed 8192
#     sbatch --time=24:0:0 --account=aip-glaj run.sh openai/gpt-oss-20b lcb 4 16 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 1 1 10 $seed

#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 2 2 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 2 4 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 2 8 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 2 16 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 2 32 10 $seed

#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 3 3 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 3 4 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 3 8 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 3 16 10 $seed
    # sbatch --time=12:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 3 32 10 $seed

#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 4 4 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 4 8 10 $seed
#     sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 4 16 10 $seed
    # sbatch --time=24:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Thinking-2507 lcb 4 16 10 $seed 32768
    # sbatch --time=24:0:0 --account=aip-glaj run_code_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 lcb 4 16 10 $seed 16384
    # sbatch --time=12:0:0 --account=aip-glaj run.sh Qwen/Qwen3-4B-Instruct-2507 lcb 4 32 10 $seed
# done

# for seed in {1234..1237}; do
#     sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Instruct-2507 aime25 1 160 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Instruct-2507 hmmt25 1 160 1 $seed
    # sbatch --time=12:0:0 run_math_gpt.sh openai/gpt-oss-20b games 1 160 1 $seed medium
    # sbatch --time=6:0:0 run_math_gpt.sh openai/gpt-oss-20b cognition 1 160 1 $seed medium
    # sbatch --time=12:0:0 run_math_gpt.sh openai/gpt-oss-20b games 4 16 10 $seed medium
    # sbatch --time=6:0:0 run_math_gpt.sh openai/gpt-oss-20b cognition 4 16 10 $seed medium
    # sbatch --time=24:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 games 1 160 1 $seed
    # sbatch --time=12:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 cognition 1 160 1 $seed
    # sbatch --time=6:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 games 4 16 10 $seed
    # sbatch --time=3:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 cognition 1 160 1 $seed
    # sbatch --time=6:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 aime25 4 16 10 $seed
    # sbatch --time=6:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 hmmt25 4 16 10 $seed
    # sbatch --time=12:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 aime25 1 160 10 $seed
    # sbatch --time=12:0:0 run_math_nemo.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 hmmt25 1 160 10 $seed
# done

# for seed in 1234 1235 1236 1237; do
#     for cl in 2048 4096 8192 16384 32768; do
#         sbatch --time=3:0:0 run_math_cl.sh Qwen/Qwen3-4B-Instruct-2507 aime25 $seed $cl
#         # sbatch --time=3:0:0 run_math_cl.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 $seed 
#     done
# done

# for seed in 1234 1235 1236 1237; do
    # sbatch --time=3:0:0 run_math.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 aime25 4 32 10 $seed
    # sbatch --time=3:0:0 run_math.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2 hmmt25 4 32 10 $seed
    # sbatch --account=aip-glaj --time=6:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 4 16 10 $seed 

    # sbatch --time=24:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 aime25 4 16 10 $seed
    # sbatch --time=24:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 hmmt25 4 16 10 $seed
    # sbatch --time=24:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 aime25 1 160 1 $seed
    # sbatch --time=24:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 hmmt25 1 160 1 $seed
    # sbatch --time=3:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 32 10 $seed low
    # sbatch --time=6:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 32 10 $seed medium
    # sbatch --time=12:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 32 10 $seed high
    # sbatch --time=3:0:0 run_math_gpt.sh openai/gpt-oss-20b aime25 4 16 10 $seed low
    # sbatch --time=3:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 16 10 $seed low
    # sbatch --time=6:0:0 run_math_gpt.sh openai/gpt-oss-20b aime25 1 160 10 $seed medium
    # sbatch --time=6:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 1 160 10 $seed medium
    # sbatch --time=12:0:0 run_math_gpt.sh openai/gpt-oss-20b aime25 4 16 10 $seed high
    # sbatch --time=12:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 16 10 $seed high

    # sbatch --time=12:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Instruct-2507 aime25 4 32 10 $seed
    # sbatch --time=12:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Thinking-2507 aime25 4 32 10 $seed
# done