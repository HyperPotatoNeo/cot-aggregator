import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import reasoning_gym
from datasets import Dataset

# Default output path (can be overridden via CLI).
OUTPUT_PATH = Path("/pscratch/sd/m/mokshjn/eval/code/")
INSTRUCTION = "Let's think step by step and output the final answer between <answer> and </answer> tags."

# CATEGORY = "algorithmic"
# TASK_LIST = [
#     "ab", "graph_color", "number_sorting", "spiral_matrix",
#     "base_conversion", "group_anagrams", "palindrome_generation", "string_insertion",
#     "binary_alternation",	 "palindrome_partitioning",  "string_manipulation",
#     "binary_matrix", "isomorphic_strings", "pool_matrix", "string_splitting",
#     "caesar_cipher", "jugs", "ransom_note", "string_synthesis",
#     "count_primes",	"letter_counting",	"rotate_matrix", "word_ladder",
#     "cryptarithm", "letter_jumble",	"rotten_oranges", "word_sequence_reversal",
#     "game_of_life_halting", "manipulate_matrix", "sentence_reordering", "word_sorting",
#     "game_of_life",	"number_filtering", "spell_backward"
# ]

# CATEGORY = "cognition"
# TASK_LIST = [
#     "color_cube_rotation", "number_sequence", "rubiks_cube", "figlet_font",
#     "modulo_grid", "needle_haystack", "rectangle_count", "arc_agi", "arc_1d"
# ]

# CATEGORY = "games"
# TASK_LIST = [
#     "boxnet", "emoji_mystery", "kakurasu", "maze", "puzzle24", "sudoku", "tsumego",
#     "futoshiki", "knight_swap", "mini_sudoku", "rush_hour", "survo", "countdown",
#     "mahjong_puzzle", "n_queens", "sokoban", "tower_of_hanoi"
# ]

# CATEGORY = "logic"
# TASK_LIST = [
#     "aiw",
#     "knights_knaves",
#     "self_reference",
#     "zebra_puzzles",
#     "circuit_logic",
#     "propositional_logic",
#     "syllogism",
# ]

# CATEGORY = "graphs"
# TASK_LIST = [
#     "course_schedule",
#     "family_relationships",
#     "largest_island",
#     "quantum_lock",
#     "shortest_path",
# ]

CATEGORY = "code"
TASK_LIST = ["bf", "codeio"]

def build_mixed_dataset(tasks: List[str], size: int, seed: int, task_kwargs: Optional[Dict[str, Dict]] = None) -> Dataset:
    """Generate a mixed dataset of total *size* by uniformly sampling tasks.

    Sampling strategy: repeatedly choose a task uniformly at random from *tasks*,
    generate one sample for that task, and continue until *size* samples are collected.
    """
    rng = random.Random(seed)

    converted = []
    while len(converted) < size:
        task = rng.choice(tasks)
        # Use a different seed per-sample to avoid duplicates from fixed-seed generators
        sample_seed = rng.randrange(1 << 30)
        # Pick kwargs only for this task (optionally merged with __default__)
        kwargs_for_task = {}
        if task_kwargs is not None:
            kwargs_for_task = task_kwargs.get(task, {})
        samples = reasoning_gym.create_dataset(task, size=1, seed=sample_seed, **kwargs_for_task)

        entry = samples[0]
        prompt = entry["question"] + " " + INSTRUCTION
        answer_str = str(entry["answer"])  # enforce consistent string type across tasks
        entry_json = json.dumps(entry, ensure_ascii=False)  # serialize heterogeneous metadata
        verification_info = {
            "dataset_name": task,
            "entry": entry_json,
            "ground_truth": answer_str,
        }
        converted.append(
            {
                "data_source": "reasoning_gym",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": CATEGORY,
                "reward_model": {"style": "rule", "ground_truth": answer_str},
                "extra_info": verification_info,
            }
        )
    return Dataset.from_list(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mixed Genesys dataset from multiple Reasoning-Gym tasks")
    parser.add_argument("--size", type=int, default=100, help="Total number of samples across all tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task sampling and generators")
    parser.add_argument("--out", type=str, default=str(OUTPUT_PATH), help="Directory to save the mixed dataset")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON mapping of task -> kwargs (with optional __default__) to pass to create_dataset")
    args = parser.parse_args()

    config_map: Optional[Dict[str, Dict]] = None
    if args.config is not None:
        with open(args.config, "r") as f:
            config_map = json.load(f)

    ds = build_mixed_dataset(TASK_LIST, args.size, args.seed, task_kwargs=config_map)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = out_dir / "eval.parquet"
    ds.to_parquet(str(parquet_file))
    print(f"Saved {len(ds):,} mixed problems to {parquet_file.resolve()}")


if __name__ == "__main__":
    main()


