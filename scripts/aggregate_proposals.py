#!/usr/bin/env python3
import argparse, os, json
from typing import List, Dict, Any, Optional
import pandas as pd

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv


# --------------------- helpers ---------------------
def extract_question_from_prompt(prompt_cell: Any) -> str:
    """
    Supports a list of chat messages like:
      [{"role": "user", "content": "..."}]
    or a raw string. Returns the first user content when list[dict].
    """
    #if isinstance(prompt_cell, str):
    #    return prompt_cell
    #if isinstance(prompt_cell, list) and prompt_cell and isinstance(prompt_cell[0], dict):
        # first user message content (matches earlier usage q[0]['content'])
    return prompt_cell[0].get("content", "")
    #raise ValueError("Unsupported prompt format; expected list[dict] or str.")


def extract_ground_truth(cell: Any) -> str:
    """
    Your earlier dataset kept GT under df_in['reward_model']['ground_truth'].
    Handle dict or (rare) list-of-dicts; otherwise fail clearly.
    """
    return cell["ground_truth"]


def build_aggregator_user_content(question: str, candidate_answers: List[str]) -> str:
    parts = []
    parts.append(
        "You are given a math problem and several candidate solutions. "
        "Some candidates may be incorrect or contain errors. "
        "Aggregate the useful ideas and produce a single, high-quality solution. "
        "Be concise and correct; if candidates disagree, choose the correct path. "
        "End with the final result in \\boxed{...}.\n"
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solutions (may contain mistakes):\n")
    for i, ans in enumerate(candidate_answers, 1):
        ans_str = (ans or "").strip()
        parts.append(f"---- Solution {i} ----\n{ans_str}\n")
    parts.append(
        "Now write a single improved solution. Provide clear reasoning and end with the final answer in \\boxed{...}."
    )
    return "\n".join(parts)


def make_chat_messages(user_content: str) -> List[Dict[str, str]]:
    # Only a user turn to be robust across chat templates
    return [{"role": "user", "content": user_content}]


def render_with_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    return {
        "pred_accuracies": [int(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
    }


# --------------------- pipeline ---------------------
def run(
    input_parquet: str,
    output_parquet: str,
    model_name: str,
    k: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    tp_size: int,
    dtype: str,
    seed: int,
):
    # Load prior results (must contain prompt, pred_answers, reward_model)
    df_in = pd.read_parquet(input_parquet)
    required_cols = ["orig_prompt", "pred_answers", "reward_model"]
    for c in required_cols:
        if c not in df_in.columns:
            raise ValueError(f"Input parquet must contain '{c}' column.")

    # Tokenizer & vLLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        trust_remote_code=True,
        seed=seed,
    )
    sampling = SamplingParams(n=k, temperature=temperature, max_tokens=max_new_tokens)

    # Build aggregator prompts (keep old prompt intact; save new under 'agg_prompt')
    agg_chat_messages: List[List[Dict[str, str]]] = []
    rendered_prompts: List[str] = []

    questions: List[str] = [extract_question_from_prompt(df_in.loc[i, "orig_prompt"]) for i in range(len(df_in))]
    candidates: List[List[str]] = [list(df_in.loc[i, "pred_answers"]) for i in range(len(df_in))]
    gts: List[str] = [extract_ground_truth(df_in.loc[i, "reward_model"]) for i in range(len(df_in))]

    for q, cand in zip(questions, candidates):
        user_content = build_aggregator_user_content(q, cand)
        msgs = make_chat_messages(user_content)
        agg_chat_messages.append(msgs)
        rendered_prompts.append(render_with_template(tokenizer, msgs))

    # Generate K improved answers per new aggregator prompt
    new_pred_answers: List[List[str]] = []
    for start in range(0, len(rendered_prompts), batch_size):
        outs = llm.generate(rendered_prompts[start : start + batch_size], sampling)
        for out in outs:
            new_pred_answers.append([o.text for o in out.outputs])

    if len(new_pred_answers) != len(df_in):
        raise RuntimeError("Generation count mismatch: got {} answers for {} rows."
                           .format(len(new_pred_answers), len(df_in)))

    # Evaluate the new answers against GT per row
    new_pred_accuracies: List[List[int]] = []
    new_mean_acc: List[float] = []
    new_pass_at_k: List[float] = []
    for k_answers, gt in zip(new_pred_answers, gts):
        eval_out = evaluate_k_answers(k_answers, gt)
        new_pred_accuracies.append(eval_out["pred_accuracies"])
        new_mean_acc.append(eval_out["mean_acc"])
        new_pass_at_k.append(eval_out["pass_at_k"])

    # Prepare output:
    # - keep original 'prompt'
    # - add 'agg_prompt' (chat messages used to form the aggregator request)
    # - overwrite 'pred_answers' with new K answers
    # - overwrite 'pred_accuracies', 'mean_acc', 'pass_at_k'
    df_out = df_in.copy()
    df_out["orig_prompt"] = df_out["orig_prompt"].copy()
    df_out["prompt"] = agg_chat_messages
    df_out["pred_answers"] = new_pred_answers
    df_out["pred_accuracies"] = new_pred_accuracies
    df_out["mean_acc"] = new_mean_acc
    df_out["pass_at_k"] = new_pass_at_k

    # Save parquet
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    df_out.to_parquet(output_parquet, index=False)
    print("Saved aggregated proposals to:", output_parquet)

    # Summary
    print(json.dumps(
        {
            "n_samples": len(df_out),
            "k": k,
            "mean_acc_k": float(sum(new_mean_acc) / max(1, len(new_mean_acc))),
            "mean_pass_at_k": float(sum(new_pass_at_k) / max(1, len(new_pass_at_k))),
            "note": "kept original prompt; added agg_prompt; recomputed and replaced metrics for new answers",
        },
        indent=2,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/pscratch/sd/s/siddart2/data/math/llama_3_3b/baseline_test.parquet")
    ap.add_argument("--output", default="/pscratch/sd/s/siddart2/data/math/llama_3_3b/baseline_agg_1_test.parquet")
    ap.add_argument("--model", default="/pscratch/sd/s/siddart2/checkpoints/verl_rloo_example_gsm8k/llama_3b_base/baseline")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto","float16","bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    run(
        input_parquet=args.input,
        output_parquet=args.output,
        model_name=args.model,
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
