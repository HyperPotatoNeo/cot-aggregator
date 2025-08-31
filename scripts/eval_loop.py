from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv
import numpy as np
import random

# --------------------- helpers ---------------------
def _append_metrics_to_json(path: str, entry: dict):
    """Append `entry` to a JSON array file at `path` (create if needed)."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                # If somehow not a list, wrap it
                data = [data]
        else:
            data = []
    except Exception:
        # Corrupt or empty file -> start fresh
        data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_question_from_prompt(prompt_cell: Any) -> str:
    """
    Supports a list of chat messages like:
      [{"role": "user", "content": "..."}]
    or a raw string. Returns the first user content when list[dict].
    """
    return prompt_cell[0].get("content", "")

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def make_chat_message(question: str) -> str:
    messages = [
        {"role": "user", "content": question},
    ]
    return messages

def make_chat_prompt(tokenizer: AutoTokenizer, messages: list[Dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def render_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    chat_message = make_chat_message(prompt)
    return make_chat_prompt(tokenizer, chat_message), chat_message

def aggregate_prompt(question: str, candidate_answers: List[str]) -> str:
    parts = []
    parts.append(
        "You are given a math problem and several candidate solutions. "
        "Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy."#"Be concise and correct; if candidates disagree, choose the correct path. "
        "End with the final result in \\boxed{{}}.\n"
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solutions (may contain mistakes):\n")
    for i, ans in enumerate(candidate_answers, 1):
        ans_str = (ans or "").strip()
        parts.append(f"---- Solution {i} ----\n{ans_str}\n")
    parts.append(
        "Now write a single improved solution. Provide clear reasoning and end with the final answer in \\boxed{{}}."
    )
    return "\n".join(parts)

def build_prompt(tokenizer: AutoTokenizer, question: str, candidate_answers: List[str]):
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers)
    else:
        prompt = question
        
    return render_chat_template(tokenizer, prompt)

def majority_vote_from_answers(k_answers: List[str], gt: str) -> Dict[str, Any]:
    """
    Cluster k answers by pairwise equality (using is_equiv on de-boxed strings),
    pick the largest cluster (majority), and compare its representative to gt.
    Tie-breaker: first cluster to reach the max size.
    """
    # Normalize to extracted final answers (no \boxed{ } wrapper)
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_boxed(s) for s in solutions]

    clusters: List[Dict[str, Any]] = []  # each: {"rep": str, "count": int}
    for e in extracted:
        placed = False
        for c in clusters:
            # Use your is_equiv for pairwise equality between answers
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        return 0

    best = max(clusters, key=lambda c: c["count"])
    mv_correct = int(bool(is_equiv(best["rep"], gt)))
    return mv_correct

def evaluate_k_answers(k_answers: List[str], gt: str) -> Dict[str, Any]:
    """
    Compute per-rollout correctness, mean accuracy, and pass@k against the ground truth.
    Uses the same boxed-extraction logic as your original script.
    """
    # Extract \boxed{...} (or dummy boxed if missing) -> remove_boxed -> is_equiv
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_boxed(s) for s in solutions]
    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]

    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)
    majority_vote = majority_vote_from_answers(k_answers, gt)
    return {
        "pred_accuracies": [int(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }

def generate_candidates(A, M, R):
    if A is None:
        return [None for _ in range(M)]

    return [random.sample(A, R) for _ in range(M)]

def reshape_list(lst, K):
    return [lst[i:i+K] for i in range(0, len(lst), K)]

def run(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling: SamplingParams,
    k: int,
    population: int,
    data: List
):

    requests, ground_truths = [], []
    for problem in data:
        prompt = problem['orig_prompt']
        ground_truth = problem['ground_truth']
        candidate_answers = generate_candidates(problem['candidates'], population, k)
        ground_truths.append(ground_truth)
        for candidates in candidate_answers:
            request, _ = build_prompt(tokenizer, prompt, candidates)
            requests.append(request)
    
    #print(requests[0])
    outs = llm.generate(requests, sampling)
    all_responses = [o.text for out in outs for o in out.outputs]
    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses

    # Evaluate
    pred_accuracies: List[List[int]] = []
    mean_acc: List[float] = []
    pass_at_k: List[int] = []
    majority_acc: List[int] = []

    for gt, responses in zip(ground_truths, all_responses):
        perf_metric = evaluate_k_answers(responses[:], gt)
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
        majority_acc.append(perf_metric['majority_vote_correct'])
    
    metrics = json.dumps(
        {
            "n_samples": len(mean_acc),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
            "mean_majority_acc": sum(majority_acc) / max(1, len(majority_acc)),
        }, indent=2
    )
    return data, metrics

def loop(
    model_name: str,
    loops: int,
    k: int,
    population: int,
    seed_dataset: str,
    output_dir: str,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
    num_seeds: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                  dtype=dtype, trust_remote_code=True, seed=seed)
    sampling = SamplingParams(
        n=1, temperature=temperature, max_tokens=max_new_tokens
    )
    df = pd.read_parquet(seed_dataset)
    
    # --- seed aggregation (added) ---
    acc_mean_acc_k = [[] for _ in range(loops)]
    acc_mean_pass_at_k = [[] for _ in range(loops)]
    acc_mean_majority_acc = [[] for _ in range(loops)]
    n_samples_record = None

    for s in range(num_seeds):
        # control RNG for candidate sampling too
        random.seed(seed + s)
        np.random.seed(seed + s)
        base_structure = [
            {
                'orig_prompt': extract_question_from_prompt(row['prompt']),
                'ground_truth': row['reward_model']['ground_truth'],
                'candidates': None,
            }
            for _, row in df.iterrows()
        ]

        for loop_idx in range(loops):
            data, metrics = run(llm, tokenizer, sampling, k, population, base_structure)
            print(metrics)
            metrics_dict = json.loads(metrics)
            if n_samples_record is None:
                n_samples_record = metrics_dict.get("n_samples", None)
            acc_mean_acc_k[loop_idx].append(metrics_dict["mean_acc_k"])
            acc_mean_pass_at_k[loop_idx].append(metrics_dict["mean_pass_at_k"])
            acc_mean_majority_acc[loop_idx].append(metrics_dict["mean_majority_acc"])

    # write aggregated per-loop metrics (lists + mean/std), path unchanged
    os.makedirs(os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population)), exist_ok=True)
    metrics_path = os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population), f'eval_loop.json')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    for loop_idx in range(loops):
        vals_acc = acc_mean_acc_k[loop_idx]
        vals_pas = acc_mean_pass_at_k[loop_idx]
        vals_maj = acc_mean_majority_acc[loop_idx]

        out_entry = {
            "n_samples": n_samples_record if n_samples_record is not None else 0,
            "k": k,
            "population": population,
            "loop": loop_idx,
            "n_seeds": num_seeds,
            "values": {
                "mean_acc_k": vals_acc,
                "mean_pass_at_k": vals_pas,
                "mean_majority_acc": vals_maj
            },
            "summary": {
                "mean_acc_k": {"mean": float(np.mean(vals_acc)), "std": float(np.std(vals_acc, ddof=0))},
                "mean_pass_at_k": {"mean": float(np.mean(vals_pas)), "std": float(np.std(vals_pas, ddof=0))},
                "mean_majority_acc": {"mean": float(np.mean(vals_maj)), "std": float(np.std(vals_maj, ddof=0))}
            }
        }
        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="/pscratch/sd/s/siddart2/data/aime/train.parquet")
    ap.add_argument("--output", default="/pscratch/sd/s/siddart2/evals/aime/ref")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-seeds", type=int, default=4)
    args = ap.parse_args()

    loop(
        model_name=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=args.output,
        k=args.k,
        population=args.population,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
        num_seeds=args.num_seeds,
    )

if __name__ == "__main__":
    main()
