from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from typing import List, Dict, Any
import argparse, json, math, os, re
from typing import List, Optional
import pandas as pd
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv

# --------------------- helpers ---------------------
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
        "Be concise and correct; if candidates disagree, choose the correct path. "
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


def run(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling: SamplingParams,
    k: int,
    df: pd.DataFrame,
    dataset_path: str,
    output_path: str,
):
    prompts = [
        row for row in df['orig_prompt']
    ]

    candidates: List[List[str]] = [
        row['pred_answers'] if 'pred_answers' in row else None for _, row in df.iterrows()
    ]

    ground_truths = [
        row['ground_truth'] for row in df['reward_model']
    ]
        
    requests, chat_messages = [], []
    
    for prompt, candidate_answers in zip(prompts, candidates):
        request, chat_msg = build_prompt(tokenizer, prompt, candidate_answers)
        requests.append(request)
        chat_messages.append(chat_msg)
    
    print(requests[0])
    outs = llm.generate(requests, sampling)
    all_responses = [[o.text for o in out.outputs] for out in outs]

    # Evaluate
    pred_answers: List[List[str]] = []
    pred_accuracies: List[List[int]] = []
    mean_acc: List[float] = []
    pass_at_k: List[int] = []
    majority_acc: List[int] = []

    for gt, responses in zip(ground_truths, all_responses):
        perf_metric = evaluate_k_answers(responses, gt)
        pred_answers.append(responses)
        pred_accuracies.append(perf_metric['pred_accuracies'])
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
        majority_acc.append(perf_metric['majority_vote_correct'])

    df_out = df.copy()
    df_out['prompt'] = chat_messages
    df_out["pred_answers"] = pred_answers                # list[str], matches GT display style
    df_out["pred_accuracies"] = pred_accuracies          # list[int] of 0/1 per rollout
    df_out["mean_acc"] = mean_acc                        # optional scalar summary
    df_out["pass_at_k"] = pass_at_k                      # optional scalar summary
    df_out["majority_acc"] = majority_acc        # str, majority vote answer
    
    metrics = json.dumps(
        {
            "n_samples": len(df_out),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
            "mean_majority_acc": sum(majority_acc) / max(1, len(majority_acc)),
        }, indent=2
    )
    print(metrics)
    #print(df_out)
    return df_out, metrics

def loop(
    model_name: str,
    loops: int,
    k: int,
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
        n=k, temperature=temperature, max_tokens=max_new_tokens
    )
    
    df = pd.read_parquet(seed_dataset)
    original_prompts = [extract_question_from_prompt(row) for row in df['prompt']]
    df['orig_prompt'] = original_prompts
    
    #print(df)
    
    for loop in range(loops):
        # df, metrics = run(None, None, None, k, df, seed_dataset, output_dir)
        df, metrics = run(llm, tokenizer, sampling, k, df, seed_dataset, output_dir)
        # DUMP DF AND METRICS
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(os.path.join(output_dir, f'agg_{loop}.parquet'), index=False)
        #df.to_parquet(os.path.join(output_dir, f'majority_16.parquet'), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset", default="/pscratch/sd/s/siddart2/data/math/test.parquet")
    ap.add_argument("--output", default="/pscratch/sd/s/siddart2/data/math/ref/")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    loop(
        model_name=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=args.output,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tp_size=args.tp_size,
        dtype=args.dtype,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
