import re

import io
import contextlib
import traceback
import multiprocessing as mp
import resource

patterns = [r"‘‘‘python(.*?)‘‘‘", r"‘‘‘(.*?)‘‘‘", r"```python(.*?)```", r"```(.*?)```"]


def _worker(code, test_cases, mem_limit_mb, conn):
    results = []
    ns = {}

    if mem_limit_mb is not None:
        bytes_limit = mem_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))

    f_stdout, f_stderr = io.StringIO(), io.StringIO()

    with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
        try:
            exec(code, ns)
        except Exception:
            error_msg = traceback.format_exc()
            for test in test_cases:
                results.append({"test": test, "passed": False, "reward": 0., "error": f"Code error: {error_msg}"})
            conn.send(results)
            conn.close()
            return

        for test in test_cases:
            try:
                exec(test, ns)
                results.append({"test": test, "passed": True, "reward": 1., "error": None})
            except Exception:
                results.append({"test": test, "passed": False, "reward": 0., "error": traceback.format_exc()})

    conn.send(results)
    conn.close()


def run_code_with_tests(code: str, test_cases: list, timeout: int = 30, mem_limit_mb: int = 1024):
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_worker, args=(code, test_cases, mem_limit_mb, child_conn))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        return [{"test": None, "passed": False, "reward": 0., "error": "Timeout exceeded"}]

    if parent_conn.poll():
        return parent_conn.recv()
    else:
        return [{"test": None, "passed": False, "reward": 0., "error": "No output from worker"}]

def compute_score(code: str, test_cases: list, timeout: int = 30, mem_limit_mb: int = 1024, continuous=False):
    solution = code.split("```python")[-1].split("```")[0]
    results = run_code_with_tests(solution, test_cases)

    if continuous:
        return sum([x['reward'] for x in results]) / len(results)
    else:
        return 1 if sum([x['reward'] for x in results]) == len(results) else 0

def extract_function_block(code_str: str) -> str:
    """
    Extracts the substring from the first line starting with 'def'
    up to the first line containing 'return' (inclusive).
    """
    lines = code_str.splitlines()
    start_idx = None
    end_idx = None

    # Find start line
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start_idx = i
            break

    if start_idx is None:
        return ""  # no function found

    # Find end line
    for i in range(start_idx, len(lines)):
        if "return" in lines[i]:
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(lines) - 1  # till the end if no return found

    # Return substring including start and end lines
    return "\n".join(lines[start_idx:end_idx + 1])

def extract_last_triple_quote(text: str) -> str | None:
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]
    
    return extract_function_block(text)