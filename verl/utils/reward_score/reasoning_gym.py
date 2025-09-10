import re

from reasoning_gym.factory import get_score_answer_fn


# make sure to include all the possible data sources in the if-else
def get_task_name(ds: Dataset) -> str:
    data_source = ds[0]['data_source']
    if "aime" in data_source or "hmmt" in data_source or "MATH" in data_source or "DeepScaleR" in data_source:
        return "math"
    elif "reasoning_gym" in data_source:
        return "rg"
    else:
        raise ValueError(f"Unknown task: {data_source}")


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



def compute_score(solution_str, extra_info):
    """The scoring function for RG tasks.

    Args:
        solution_str: the solution text
        extra_info: verification info
    """
    dataset_name = extra_info["dataset_name"]
    entry = extra_info["entry"]

    answer = _extract_post_string(solution_str)
    if answer is None:
        return 0.0

    score_answer_fn = get_score_answer_fn(name=dataset_name)
    return score_answer_fn(answer=answer, entry=entry)



