from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from verl.utils.reward_score.prime_code import compute_score

def compute_reward(output, ground_truth):
    return compute_score(output, ground_truth)

def compute_rewards(
    outputs,
    ground_truths,
):
    max_workers = min(32, len(ground_truths))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for output, ground_truth in zip(outputs, ground_truths):
            args = (output, ground_truth)
            futures.append(executor.submit(compute_reward, *args))

    return list(future.result() for future in futures)

seed_dataset = f'data/he/test.parquet'

# Load the Data
df = pd.read_parquet(seed_dataset)
codes = df['code']
ground_truths = df['ground_truth']

print(compute_score(codes[0], ground_truths[0]))
rewards = compute_rewards(codes, ground_truths)
print(rewards)