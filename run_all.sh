#!/bin/bash

for seed in {1234..1237}; do
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 1 1 10 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 1 1 10 $seed

    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 1 40 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 1 40 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 1 80 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 1 80 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 1 160 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 1 160 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 aime25 1 320 1 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen3-4B-Instruct-2507 hmmt25 1 320 1 $seed

    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen2.5-3B-Instruct math 4 32 10 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen2.5-7B-Instruct math 4 32 10 $seed
    # sbatch --time=3:0:0 run_math.sh Qwen/Qwen2.5-32B-Instruct math 4 32 10 $seed
    sbatch --time=9:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 aime25 4 32 10 $seed
    sbatch --time=9:0:0 run_math_thinking.sh Qwen/Qwen3-4B-Thinking-2507 hmmt25 4 32 10 $seed
    # sbatch --time=3:0:0 run_math_gpt.sh openai/gpt-oss-20b hmmt25 4 32 10 $seed
    # sbatch --time=12:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Instruct-2507 aime25 4 32 10 $seed
    # sbatch --time=12:0:0 run_math.sh Qwen/Qwen3-30B-A3B-Thinking-2507 aime25 4 32 10 $seed
done