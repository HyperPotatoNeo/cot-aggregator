import argparse
import json
from pathlib import Path

import reasoning_gym
from datasets import Dataset

# Default output path (can be overridden via CLI).
OUTPUT_PATH = Path("/pscratch/sd/m/mokshjn/eval/")
INSTRUCTION = "Let's think step by step and output the final answer between <answer> and </answer> tags."

def build_dataset(task: str, size: int, seed: int) -> Dataset:
    """Generate *size* samples from *task* and return HF Dataset."""
    samples = reasoning_gym.create_dataset(task, size=size, seed=seed)  # type: ignore[attr-defined]

    converted = []
    for idx, entry in enumerate(samples):
        prompt = entry["question"] + " " + INSTRUCTION
        verification_info = {
            "dataset_name": task,
            "entry": entry,
            "ground_truth": entry["answer"],
        }
        converted.append(
            {
                "data_source": "reasoning_gym",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": entry["answer"]},
                "extra_info": verification_info,
            }
        )

    return Dataset.from_list(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Genesys dataset from Reasoning-Gym task")
    parser.add_argument("--task", type=str, default="mini_sudoku", help="Reasoning-Gym task name (e.g. mini_sudoku, arc_agi, …)")
    parser.add_argument("--size", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RG data generator")
    parser.add_argument("--out", type=str, default=str(OUTPUT_PATH), help="Path to save the dataset (directory will be created)")
    args = parser.parse_args()

    ds = build_dataset(args.task, args.size, args.seed)

    out_dir = Path(args.out) / f"{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save dataset as a parquet file
    parquet_file = out_dir / f"{args.task}_test.parquet"
    ds.to_parquet(str(parquet_file))
    print(f"Saved {len(ds):,} problems to {parquet_file.resolve()}")


if __name__ == "__main__":
    main()