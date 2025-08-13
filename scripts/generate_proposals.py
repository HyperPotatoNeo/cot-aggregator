#!/usr/bin/env python3
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --------------------- helpers ---------------------
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def make_chat_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        #{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run(
    model_name: str,
    dataset_path: str,
    output_path: str,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
):
    # Load dataset
    df_in = pd.read_parquet(dataset_path)

    # Tokenizer & vLLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, tensor_parallel_size=tp_size,
              dtype=dtype, trust_remote_code=True, seed=seed)

    sampling = SamplingParams(
        n=k, temperature=temperature, max_tokens=max_new_tokens
    )

    # Build prompts
    prompts = [make_chat_prompt(tokenizer, q[0]['content']) for q in df_in['prompt']]
    gts = [d['ground_truth'] for d in df_in['reward_model']]

    # Generate
    all_responses: List[List[str]] = []
    for i in range(0, len(prompts), batch_size):
        outs = llm.generate(prompts[i:i+batch_size], sampling)
        for out in outs:
            all_responses.append([o.text for o in out.outputs])
    #print([extract_solution(last_boxed_only_string(t)) for t in all_responses[0]])

    # Evaluate
    pred_answers: List[List[str]] = []
    pred_accuracies: List[List[int]] = []
    mean_acc: List[float] = []
    pass_at_k: List[int] = []

    for i, responses in enumerate(all_responses):
        solutions = [last_boxed_only_string(t) if last_boxed_only_string(t) is not None else "\\boxed{}" for t in responses]
        extracted = [remove_boxed(t) for t in solutions]
        correct_bools = [is_equiv(e, gts[i]) for e in extracted]
        pred_answers.append(responses)
        pred_accuracies.append([int(b) for b in correct_bools])
        acc_mean = sum(correct_bools) / max(1, len(correct_bools))
        mean_acc.append(float(acc_mean))
        pass_at_k.append(1.0 if any(correct_bools) else 0.0)

    # Write: keep original columns, add the two new list columns (and summary metrics if you want)
    df_out = df_in.copy()
    df_out["pred_answers"] = pred_answers     # list[str], matches GT display style
    df_out["orig_prompt"] = df_out["prompt"]  # keep original prompt
    df_out["pred_accuracies"] = pred_accuracies          # list[int] of 0/1 per rollout
    df_out["mean_acc"] = mean_acc                        # optional scalar summary
    df_out["pass_at_k"] = pass_at_k                      # optional scalar summary

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    # quick aggregate
    print(json.dumps({
        "n_samples": len(df_out),
        "k": k,
        "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
        "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
    }, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="/pscratch/sd/s/siddart2/data/math/test.parquet")
    ap.add_argument("--output", default="/pscratch/sd/s/siddart2/data/math/test_qwen3_K.parquet")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    run(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        k=args.k,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
