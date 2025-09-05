from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from verl.utils.reward_score.code import compute_score
import numpy as np
import random

instruction_following = ("Please follow the following instructions:\n\n"
                "- Reason about the problem and any base cases before writing the code.\n"
                "- You must return the implementation code in the following format:\n"
                "```python\n"
                "<CODE GOES HERE>\n"
                "```\n"
                # "- You must only return a single code block since we only parse the first code block.\n"
                "- Do not include any tests in your code - we will run the suite and return any error feedback.\n"
                "- Include relevant import statements.\n"
            )


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
        "You are given a python code implementation problem and several candidate code blocks with their reasoning. "
        "Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Reason carefully; if candidates disagree, choose the correct path."
        # "Be concise and correct; if candidates disagree, choose the correct path. "
    )
    # parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solutions (may contain mistakes):\n")
    for i, ans in enumerate(candidate_answers, 1):
        ans_str = (ans or "").strip()
        parts.append(f"---- Solution {i} ----\n{ans_str}\n")
    parts.append(
        "\nNow provide an improved and correct solution along with its reasoning. " + instruction_following
    )
    return "\n".join(parts)

def build_prompt(tokenizer: AutoTokenizer, question: str, candidate_answers: List[str]):
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers)
    else:
        prompt = question + "\n\n" + instruction_following
        
    return render_chat_template(tokenizer, prompt)

def evaluate_k_answers(k_answers: List[str], gt: str) -> Dict[str, Any]:
    """
    Compute per-rollout correctness, mean accuracy, and pass@k against the ground truth.
    Uses the same boxed-extraction logic as your original script.
    """
    correct_bools = [compute_score(e, gt, continuous=False) for e in k_answers]

    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)
    return {
        "pred_accuracies": [int(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
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
    data: List,
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
    
    print(requests[0])
    outs = llm.generate(requests, sampling)
    all_responses = [o.text for out in outs for o in out.outputs]
    sequence_lengths = [len(o.token_ids) for out in outs for o in out.outputs]
    mean_seq_len = sum(sequence_lengths) / len(sequence_lengths)
    max_seq_len = max(sequence_lengths)
    print(all_responses[0])

    print(f'Mean Sequence Length: {mean_seq_len} | Max Sequence Length: {max_seq_len}')

    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses

    # Evaluate
    mean_acc: List[float] = []
    pass_at_k: List[int] = []

    max_workers = min(48, len(ground_truths))
    perf_metrics = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for gt, responses in zip(ground_truths, all_responses):
            args = (responses, gt)
            perf_metrics.append(executor.submit(evaluate_k_answers, *args))

    perf_metrics = list(perf_metric.result() for perf_metric in perf_metrics)

    for perf_metric in perf_metrics:
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
    
    print(mean_acc)
    metrics = json.dumps(
        {
            "n_samples": len(mean_acc),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
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

        data = [
            {
                'orig_prompt': extract_question_from_prompt(row['prompt']),
                'ground_truth': row['test_cases'],
                'candidates': None,
            }
            for _, row in df.iterrows()
        ]
        
        for loop_idx in range(loops):
            data, metrics = run(llm, tokenizer, sampling, k, population, data)
            print(metrics)
            metrics_dict = json.loads(metrics)
            if n_samples_record is None:
                n_samples_record = metrics_dict.get("n_samples", None)
            acc_mean_acc_k[loop_idx].append(metrics_dict["mean_acc_k"])
            acc_mean_pass_at_k[loop_idx].append(metrics_dict["mean_pass_at_k"])

    # write aggregated per-loop metrics (lists + mean/std), path unchanged
    os.makedirs(os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population)), exist_ok=True)
    metrics_path = os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population), f'eval_loop.json')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    for loop_idx in range(loops):
        vals_acc = acc_mean_acc_k[loop_idx]
        vals_pas = acc_mean_pass_at_k[loop_idx]

        out_entry = {
            "n_samples": n_samples_record if n_samples_record is not None else 0,
            "k": k,
            "population": population,
            "loop": loop_idx,
            "n_seeds": num_seeds,
            "values": {
                "mean_acc_k": vals_acc,
                "mean_pass_at_k": vals_pas,
            },
            "summary": {
                "mean_acc_k": {"mean": float(np.mean(vals_acc)), "std": float(np.std(vals_acc, ddof=0))},
                "mean_pass_at_k": {"mean": float(np.mean(vals_pas)), "std": float(np.std(vals_pas, ddof=0))},
            }
        }
        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="./data/he/test.parquet")
    ap.add_argument("--output", default="./data/he/ref")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
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
        num_seeds=args.num_seeds
    )

if __name__ == "__main__":
    main()
