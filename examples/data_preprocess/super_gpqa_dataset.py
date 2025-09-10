import argparse
import os
from typing import List

import datasets
import pandas as pd

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def build_user_message(question: str, choices: List[str], instruction: str) -> str:
    lines = [question.strip(), ""]
    for i, opt in enumerate(choices):
        lines.append(f"Option {LETTERS[i]} : {opt}")
        lines.append("")  # blank line between options
    lines.append(instruction.strip())
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/pscratch/sd/s/siddart2/data/supergpqa/")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--ability", default="supergpqa")
    parser.add_argument(
        "--instruction",
        default="Think step by step and output the final correct option letter within \\boxed{}; for example \\boxed{A}.",
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print("Loading m-a-p/SuperGPQA...")
    ds_dict = datasets.load_dataset("m-a-p/SuperGPQA")
    base = ds_dict[args.split]

    def process_fn(example, idx):
        content = build_user_message(example["question"], example["options"], args.instruction)
        return {
            "data_source": "m-a-p/SuperGPQA",
            "prompt": [{"role": "user", "content": content}],
            "ability": args.ability,
            "reward_model": {"style": "rule", "ground_truth": example['answer_letter']},
            "extra_info": {
                "split": args.split,
                "index": idx,
                "uuid": example.get("uuid"),
                "discipline": example.get("discipline"),
                "field": example.get("field"),
                "subfield": example.get("subfield"),
                "difficulty": example.get("difficulty"),
                "is_calculation": bool(example.get("is_calculation", False)),
            },
            "subject": example.get("subfield"),
        }

    out_ds = base.map(function=process_fn, with_indices=True, remove_columns=base.column_names)

    out_path = os.path.join(args.local_dir, f"supergpqa_{args.split}.parquet")
    out_ds.to_parquet(out_path)
    print(f"Saved parquet to {out_path}")

    pdf = out_ds.to_pandas()
    print(pdf["subject"].value_counts().sort_index())


if __name__ == "__main__":
    main()
