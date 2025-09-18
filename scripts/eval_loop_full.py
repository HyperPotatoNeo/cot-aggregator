from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional, Callable
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

def extract_rg_solution(completion: str) -> Optional[str]:
    """Extract the model's predicted answer for reasoning-gym style prompts.

    Priority order:
    1. Anything enclosed by <answer> ... </answer> (preferred new format).
    2. Text following "Final Answer:" after an optional "</think>" tag (legacy format).
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None

    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None

    return final_response[1].strip()

# make sure to include all the possible data sources in the if-else
def get_task_name(df: pd.DataFrame) -> str:
    data_source = df['data_source'][0]
    if "aime" in data_source or "hmmt" in data_source or "MATH" in data_source or "DeepScaleR" in data_source:
        return "math"
    elif "reasoning_gym" in data_source:
        return "rg"
    elif "gpqa" in data_source.lower():
        return "qa"
    else:
        raise ValueError(f"Unknown task: {data_source}")


# --------------------- prompt building ---------------------
def render_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    chat_message = [
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True), chat_message


def aggregate_prompt(question: str, candidate_answers: List[str], task: str) -> str:
    if task == 'rg':
        problem_kind = 'problem'
        format_hint = '<answer>...</answer>'
    else:
        problem_kind = 'math problem'
        format_hint = '\\boxed{}'

    parts = []
    if len(candidate_answers) == 1:
        parts.append(
            f"You are given a {problem_kind} and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            f"End with the final result in {format_hint}.\n"
        )
    else:
        parts.append(
            f"You are given a {problem_kind} and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy."
            f"End with the final result in {format_hint}.\n"
        )

    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(f"---- Candidate ----\n{ans_str}\n")
        parts.append(
            f"Now refine the candidate into an improved solution. Provide clear reasoning and end with the final answer in {format_hint}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(f"---- Solution {i} ----\n{ans_str}\n")
        parts.append(
            f"Now write a single improved solution. Provide clear reasoning and end with the final answer in {format_hint}."
        )

    return "\n".join(parts)


def build_prompt(tokenizer: AutoTokenizer, question: str, candidate_answers: Optional[List[str]], task: str):
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers, task)
    else:
        prompt = question
    return render_chat_template(tokenizer, prompt)


# --------------------- summarization ---------------------
def summarize_cot_prompt(question: str, candidate: str) -> str:
    parts = []
    parts.append(
        "You are given a math problem and a candidate solution. "
        "Summarize the solution into a concise chain-of-thought style outline that preserves all "
        "important information required to continue refinement later: the main approach(es), key steps/equations, "
        "useful intermediate results, and any mistakes or dead ends. "
        "Compress it while keeping the essential structure. "
        "If the candidate included a final answer, retain it at the end in \\boxed{ }.\n"
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solution:\n")
    parts.append(candidate.strip() + "\n")
    parts.append("Now produce the concise, information-preserving summary. "
                 "End with the final answer in \\boxed{} if present.")
    return "\n".join(parts)


def summarize_candidates_inplace(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[dict],
    max_tokens: int,
    temperature: float
) -> None:
    """
    For each problem, summarize each candidate individually and replace in place.
    """
    requests = []
    idxs = []  # (problem_idx, candidate_idx)
    for pi, problem in enumerate(data):
        question = problem['orig_prompt']
        cands = problem.get('candidates') or []
        for ci, cand in enumerate(cands):
            # Build a chat prompt per candidate
            prompt = summarize_cot_prompt(question, cand)
            chat_prompt, _ = render_chat_template(tokenizer, prompt)
            requests.append(chat_prompt)
            idxs.append((pi, ci))

    if not requests:
        return

    summarize_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens
    )
    outs = llm.generate(requests, summarize_params)
    flat = [o.text for out in outs for o in out.outputs]

    # Write summaries back in place
    for (pi, ci), summary in zip(idxs, flat):
        data[pi]['candidates'][ci] = summary


# --------------------- evaluation ---------------------
def evaluate_k_answers_math(k_answers: List[str], gt: str) -> Dict[str, Any]:
    lengths = [len(ans) for ans in k_answers]
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_boxed(s) for s in solutions]

    ## mean accuracy, pass@k
    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)
    mean_length = float(sum(lengths) / max(1, len(lengths)))

    ## majority vote
    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_vote = 0.0

    best = max(clusters, key=lambda c: c["count"])
    majority_vote = float(bool(is_equiv(best["rep"], gt)))

    return {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote,
        "mean_length": mean_length
    }

def evaluate_k_answers_qa(k_answers: List[str], gt: str) -> Dict[str, Any]:
    def remove_text_box(s: str) -> str:
        return re.sub(r'\\text\{([^}]*)\}', r'\1', s)

    lengths = [len(ans) for ans in k_answers]
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_text_box(remove_boxed(s)) for s in solutions]

    ## mean accuracy, pass@k
    correct_bools = [bool(e == gt) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)
    mean_length = float(sum(lengths) / max(1, len(lengths)))

    ## majority vote
    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_vote = 0.0

    best = max(clusters, key=lambda c: c["count"])
    majority_vote = float(bool(is_equiv(best["rep"], gt)))

    return {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote,
        "mean_length": mean_length
    }

def evaluate_k_answers_rg(score_answer_fn: Callable[[str, str], float], k_answers: List[str], gt: str) -> Dict[str, Any]:
    lengths = [len(ans) for ans in k_answers]
    solutions = [extract_rg_solution(a) or "" for a in k_answers]

    ## mean accuracy, pass@k
    scores = [float(score_answer_fn(sol, gt)) for sol in solutions]
    mean_acc = float(sum(scores) / max(1, len(scores)))
    pass_at_k = float(1.0 if any(s == 1.0 for s in scores) else 0.0)

    ## majority vote
    clusters: List[Dict[str, Any]] = []
    for sol in solutions:
        placed = False
        for c in clusters:
            if bool(is_equiv(sol, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": sol, "count": 1})

    if not clusters:
        majority_vote = 0.0

    best = max(clusters, key=lambda c: c["count"])
    majority_vote = float(score_answer_fn(best["rep"], gt))

    return {
        "pred_accuracies": [float(s) for s in scores],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


# --------------------- main ---------------------
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
    task: str,
    score_answer_fn: Optional[Callable[[str, str], float]] = None,
):

    requests, ground_truths = [], []
    for problem in data:
        prompt = problem['orig_prompt']
        ground_truth = problem['gt']
        candidate_answers = generate_candidates(problem['candidates'], population, k)
        ground_truths.append(ground_truth)
        for candidates in candidate_answers:
            request, _ = build_prompt(tokenizer, prompt, candidates, task)
            requests.append(request)
    
    #print(requests[0])
    outs = llm.generate(requests, sampling)
    all_responses = [o.text for out in outs for o in out.outputs]

    mean_response_length = [len(tokenizer.encode(response)) for response in all_responses]
    median = np.percentile(mean_response_length, 50)
    q25 = np.percentile(mean_response_length, 25)
    q75 = np.percentile(mean_response_length, 75)
    mean_response_length = sum(mean_response_length) / max(1, len(mean_response_length))

    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses

    # Evaluate
    pred_accuracies: List[List[float]] = []
    mean_acc: List[float] = []
    pass_at_k: List[float] = []
    majority_acc: List[float] = []

    for gt, responses in zip(ground_truths, all_responses):
        if task == 'rg':
            assert score_answer_fn is not None, "score_answer_fn must be provided for task 'rg'"
            perf_metric = evaluate_k_answers_rg(score_answer_fn, responses[:], gt)
        elif task == 'math':
            perf_metric = evaluate_k_answers_math(responses[:], gt)
        elif task == 'qa':
            perf_metric = evaluate_k_answers_qa(responses[:], gt)
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
            "mean_length": mean_response_length,
            "median_length": median,
            "q25_length": q25,
            "q75_length": q75,
        }, indent=2
    )
    return data, metrics


def loop(
    model_name: str,
    loops: int,
    k: int,
    population: int,
    summarize_cot: bool,
    seed_dataset: str,
    output_dir: str,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(model=model_name, tensor_parallel_size=tp_size,
                  dtype=dtype, trust_remote_code=True, seed=seed)
    sampling = SamplingParams(
        n=1, temperature=temperature, max_tokens=max_new_tokens
    )
    df = pd.read_parquet(seed_dataset)

    # Prepare scorer for RG when needed (lazy import to avoid dep when not used)
    score_answer_fn: Optional[Callable[[str, str], float]] = None
    task = get_task_name(df)
    if task == 'rg':
        from reasoning_gym.factory import get_score_answer_fn
        score_answer_fn = get_score_answer_fn(name=df['extra_info'][0]['dataset_name'])

    # --- seed aggregation (applies to both) ---
    acc_mean_acc_k = [[] for _ in range(loops)]
    acc_mean_pass_at_k = [[] for _ in range(loops)]
    acc_mean_majority_acc = [[] for _ in range(loops)]
    n_samples_record = None

    # control RNG for candidate sampling too
    random.seed(seed)
    np.random.seed(seed)
    base_structure = [
        {
            'orig_prompt': extract_question_from_prompt(row['prompt']),
            'gt': row['extra_info']['entry'] if task == 'rg' else row['reward_model']['ground_truth'],
            'candidates': None,
        }
        for _, row in df.iterrows()
    ][:1000]

    # write aggregated per-loop metrics (lists + mean/std), path unchanged
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir,'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed)+'.json')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    for loop_idx in range(loops):
        data, metrics = run(
            llm=llm,
            tokenizer=tokenizer,
            sampling=sampling,
            k=k,
            population=population,
            data=base_structure,
            task=task,
            score_answer_fn=score_answer_fn,
        )
        print(loop_idx, metrics)
        if summarize_cot and loop_idx < loops - 1:
            print("Summarizing candidates before aggregation...")
            summarize_candidates_inplace(
                llm=llm,
                tokenizer=tokenizer,
                data=base_structure,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
        metrics_dict = json.loads(metrics)
        if n_samples_record is None:
            n_samples_record = metrics_dict.get("n_samples", None)

        out_entry = {
            "n_samples": n_samples_record if n_samples_record is not None else 0,
            "k": k,
            "population": population,
            "loop": loop_idx,
            "task": task,
            "mean_acc_k": metrics_dict["mean_acc_k"],
            "mean_pass_at_k": metrics_dict["mean_pass_at_k"],
            "mean_majority_acc": metrics_dict["mean_majority_acc"],
            "mean_length": metrics_dict["mean_length"],
            "median_length": metrics_dict["median_length"],
            "q25_length": metrics_dict["q25_length"],
            "q75_length": metrics_dict["q75_length"],
        }

        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="data/gpqa/train.parquet")
    ap.add_argument("--output", default="eval/")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--summarize-cot", action="store_true")
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    loop(
        model_name=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=os.path.join(args.output, args.model.split('/')[-1]),
        k=args.k,
        population=args.population,
        summarize_cot=args.summarize_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
