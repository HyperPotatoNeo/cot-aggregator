import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_kn_from_dirname(dirname: str, pattern: re.Pattern) -> Optional[Tuple[int, int]]:
    match = pattern.fullmatch(dirname)
    if not match:
        return None
    try:
        k_value = int(match.group("k"))
        n_value = int(match.group("N"))
    except Exception:
        return None
    return k_value, n_value


def read_metrics_from_dir(dir_path: str) -> List[Dict]:
    """
    Collect loop-wise metrics from a directory.

    Supported layouts inside dir_path:
    - One file per loop: metrics_loop_{i}.json containing an object with key 'mean_acc_k'.
      The loop index is inferred from the filename.
    - A single JSON file that is an array of objects each having keys 'loop' and 'mean_acc_k'.
    - Fallback to any *.json that contains an object with 'loop' and 'mean_acc_k'.
    """
    results: List[Dict] = []

    # 1) Per-loop files
    loop_file_regex = re.compile(r"metrics_loop_(\d+)\.json$")
    try:
        entries = sorted(os.listdir(dir_path))
    except FileNotFoundError:
        return results

    loop_files = []
    for name in entries:
        m = loop_file_regex.match(name)
        if m:
            loop_index = int(m.group(1))
            loop_files.append((loop_index, os.path.join(dir_path, name)))

    # If we found per-loop files, parse them
    if loop_files:
        loop_files.sort(key=lambda t: t[0])
        for loop_index, fpath in loop_files:
            try:
                with open(fpath, "r") as f:
                    obj = json.load(f)
                # Filter by samples == 100 if present
                n_samples = obj.get("n_samples", obj.get("samples"))
                if n_samples is not None and int(n_samples) != 100:
                    continue
                mean_acc = obj.get("mean_acc_k", obj.get("mean_acc"))
                mean_pass = obj.get("mean_pass_at_k")
                if mean_acc is None and mean_pass is None:
                    continue
                rec: Dict = {"loop": loop_index}
                if mean_acc is not None:
                    rec["mean_acc_k"] = float(mean_acc)
                if mean_pass is not None:
                    rec["mean_pass_at_k"] = float(mean_pass)
                results.append(rec)
            except Exception:
                continue
        return results

    # 2) Otherwise, look for a JSON file that is an array of records
    json_files = [os.path.join(dir_path, n) for n in entries if n.endswith(".json")]
    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        # If it's an array of objects with loop + mean_acc_k
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Keep only n_samples == 100 and deduplicate by loop keeping the last occurrence
            latest_by_loop: Dict[int, Dict] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                n_samples = item.get("n_samples", item.get("samples"))
                if n_samples is not None and int(n_samples) != 100:
                    continue
                loop_index = item.get("loop")
                mean_acc = item.get("mean_acc_k", item.get("mean_acc"))
                mean_pass = item.get("mean_pass_at_k")
                if loop_index is None or (mean_acc is None and mean_pass is None):
                    continue
                try:
                    rec: Dict = {"loop": int(loop_index)}
                    if mean_acc is not None:
                        rec["mean_acc_k"] = float(mean_acc)
                    if mean_pass is not None:
                        rec["mean_pass_at_k"] = float(mean_pass)
                    latest_by_loop[int(loop_index)] = rec
                except Exception:
                    continue
            if latest_by_loop:
                local_list = sorted(latest_by_loop.values(), key=lambda r: r["loop"]) 
                return local_list

    # 3) Fallback: single-object JSONs with loop and mean_acc_k
    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                obj = json.load(f)
        except Exception:
            continue
        if isinstance(obj, dict):
            # Filter by samples == 100 if present
            n_samples = obj.get("n_samples", obj.get("samples"))
            if n_samples is not None and int(n_samples) != 100:
                continue
            loop_index = obj.get("loop")
            mean_acc = obj.get("mean_acc_k", obj.get("mean_acc"))
            mean_pass = obj.get("mean_pass_at_k")
            if loop_index is not None and (mean_acc is not None or mean_pass is not None):
                try:
                    rec: Dict = {"loop": int(loop_index)}
                    if mean_acc is not None:
                        rec["mean_acc_k"] = float(mean_acc)
                    if mean_pass is not None:
                        rec["mean_pass_at_k"] = float(mean_pass)
                    return [rec]
                except Exception:
                    pass

    return results


def collect_results(root: str, dir_pattern: str) -> pd.DataFrame:
    pattern = re.compile(dir_pattern)
    rows: List[Dict] = []

    try:
        subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    except FileNotFoundError:
        raise SystemExit(f"Root directory not found: {root}")

    for dirname in subdirs:
        kn = parse_kn_from_dirname(dirname, pattern)
        if kn is None:
            continue
        k_value, n_value = kn
        dir_path = os.path.join(root, dirname)

        metrics = read_metrics_from_dir(dir_path)
        for rec in metrics:
            rows.append({
                "k": k_value,
                "N": n_value,
                "loop": rec["loop"],
                "mean_acc_k": rec.get("mean_acc_k"),
                "mean_pass_at_k": rec.get("mean_pass_at_k"),
                "config": f"k={k_value}, N={n_value}",
                "dir": dirname,
            })

    if not rows:
        return pd.DataFrame(columns=["k", "N", "loop", "mean_acc_k", "mean_pass_at_k", "config", "dir"])  # empty
    df = pd.DataFrame(rows)
    df = df.sort_values(["k", "N", "loop"]).reset_index(drop=True)
    return df


def plot_results(df: pd.DataFrame, out_path: Optional[str]) -> None:
    if df.empty:
        print("No data found to plot.")
        return

    plt.figure(figsize=(8, 5))
    for (config, group) in df.groupby("config", sort=False):
        plt.plot(group["loop"], group["mean_acc_k"], marker="o", label=config)

    plt.xlabel("Loop")
    plt.ylabel("Mean Accuracy (mean_acc_k)")
    plt.title("Mean Accuracy vs Loops by (k, N)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Config", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


def plot_pass_minus_acc_vs_loops_for_N(df: pd.DataFrame, fixed_N: int = 16, out_path: Optional[str] = None) -> None:
    """Plot (mean_pass_at_k - mean_acc_k) vs loops for a fixed N, across different K values.

    - X-axis: loop
    - One line per k for the given N
    - Saves to out_path if provided, otherwise shows interactively
    """
    if df.empty:
        print("No data found to plot.")
        return

    sub = df[df["N"] == fixed_N].copy()
    if sub.empty:
        print(f"No rows found for N={fixed_N}.")
        return

    # Need both metrics present
    if "mean_pass_at_k" not in sub.columns or "mean_acc_k" not in sub.columns:
        print("Required columns not found in DataFrame.")
        return
    sub = sub.dropna(subset=["mean_pass_at_k", "mean_acc_k"]).copy()
    if sub.empty:
        print("No rows with both mean_pass_at_k and mean_acc_k to plot.")
        return
    sub["pass_minus_acc"] = sub["mean_pass_at_k"] - sub["mean_acc_k"]

    style_by_k = {
        1: {"color": "C0", "marker": "o"},
        2: {"color": "C0", "marker": "^"},
        3: {"color": "C1", "marker": "s"},
        4: {"color": "C2", "marker": "D"},
    }

    plt.figure(figsize=(6, 4.5))
    for k_value, group in sub.groupby("k", sort=False):
        group = group.sort_values("loop")
        k_int = int(k_value)
        style = style_by_k.get(k_int, {"color": None, "marker": "o"})
        plt.plot(
            group["loop"],
            group["pass_minus_acc"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"K={k_int}",
        )

    plt.xlabel("Loop step")
    plt.ylabel("pass@N - mean@N")
    plt.title(f"Countdown - (pass@N - mean@N) vs loops (N={fixed_N})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


def plot_pass_vs_loops_for_K(df: pd.DataFrame, fixed_K: int = 4, out_path: Optional[str] = None) -> None:
    """Plot mean_pass_at_k vs loops for a fixed K, across different N values.

    - X-axis: loop
    - One line per N for the given K
    - Saves to out_path if provided, otherwise shows interactively
    """
    if df.empty:
        print("No data found to plot.")
        return

    sub = df[df["k"] == fixed_K].copy()
    if sub.empty:
        print(f"No rows found for K={fixed_K}.")
        return

    if "mean_pass_at_k" not in sub.columns:
        print("Column mean_pass_at_k not found in DataFrame.")
        return
    sub = sub.dropna(subset=["mean_pass_at_k"]).copy()
    if sub.empty:
        print("No mean_pass_at_k values to plot for the given K.")
        return

    style_by_N = {
        4: {"color": "C0", "marker": "o"},
        8: {"color": "C1", "marker": "^"},
        16: {"color": "C2", "marker": "s"},
        32: {"color": "C3", "marker": "D"},
    }


    # Use color cycle by N
    plt.figure(figsize=(6, 4.5))
    for n_value, group in sub.groupby("N", sort=False):
        group = group.sort_values("loop")
        plt.plot(
            group["loop"],
            group["mean_pass_at_k"],
            color=style_by_N[int(n_value)]["color"],
            marker=style_by_N[int(n_value)]["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"N={int(n_value)}",
        )

    plt.xlabel("Loop step")
    plt.ylabel("pass@N")
    plt.title(f"Countdown - pass@N vs loops (K={fixed_K})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()

def plot_k_vs_loops_for_N(df: pd.DataFrame, fixed_N: int = 16, out_path: Optional[str] = None) -> None:
    """Plot mean_acc_k vs loops for a fixed N, across different K values.

    - X-axis: loop
    - One line per k for the given N
    - Saves to out_path if provided, otherwise shows interactively
    """
    if df.empty:
        print("No data found to plot.")
        return

    sub = df[df["N"] == fixed_N].copy()
    if sub.empty:
        print(f"No rows found for N={fixed_N}.")
        return

    # Style mapping per requirement
    style_by_k = {
        1: {"color": "C0", "marker": "o"},  # circle
        2: {"color": "C1", "marker": "^"},  # triangle
        3: {"color": "C2", "marker": "s"},  # square
        4: {"color": "C3", "marker": "D"},  # diamond
    }

    # Slightly wider than taller
    plt.figure(figsize=(6, 4.5))
    plotted_ks = set()

    # Ensure k=1 is always included
    if 1 not in plotted_ks:
        k1_all = df[df["k"] == 1]
        if not k1_all.empty:
            # Prefer N=16, else N=8 for the k=1 curve
            k1_candidates = k1_all.groupby("N")
            available_Ns = set(k1_candidates.groups.keys())
            preferred_N = 1 if 1 in available_Ns else None
            if preferred_N is not None:
                k1_group = k1_candidates.get_group(preferred_N).sort_values("loop")
                plt.plot(
                    k1_group["loop"],
                    k1_group["mean_acc_k"],
                    color=style_by_k[1]["color"],
                    marker=style_by_k[1]["marker"],
                    linestyle="-",
                    linewidth=2,
                    markersize=6,
                    label=f"K=1",
                )

    for k_value, group in sub.groupby("k", sort=False):
        group = group.sort_values("loop")
        k_int = int(k_value)
        style = style_by_k.get(k_int, {"color": None, "marker": "o"})
        plt.plot(
            group["loop"],
            group["mean_acc_k"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"K={k_int}",
        )
        plotted_ks.add(int(k_value))


    plt.xlabel("Loop step")
    plt.ylabel("Mean accuracy")
    plt.title(f"Countdown - Mean accuracy vs loops (N={fixed_N})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


def plot_final_acc_vs_N_by_K(df: pd.DataFrame, final_loop: int = 4, out_path: Optional[str] = None) -> None:
    """Plot final mean accuracy vs N for different values of K at a given loop.

    - X-axis: N
    - Y-axis: mean_acc_k at loop == final_loop
    - One line per K
    """
    if df.empty:
        print("No data found to plot.")
        return

    if "mean_acc_k" not in df.columns:
        print("Column mean_acc_k not found in DataFrame.")
        return

    sub = df[df["loop"] == final_loop].copy()
    if sub.empty:
        print(f"No rows found for loop={final_loop}.")
        return

    # Remove rows without accuracy
    sub = sub.dropna(subset=["mean_acc_k"]).copy()
    if sub.empty:
        print("No mean_acc_k values to plot for the given loop.")
        return

    style_by_k = {
        1: {"color": "C0", "marker": "o"},
        2: {"color": "C0", "marker": "^"},
        3: {"color": "C1", "marker": "s"},
        4: {"color": "C2", "marker": "D"},
    }

    plt.figure(figsize=(6, 4.5))
    for k_value, group in sub.groupby("k", sort=False):
        k_int = int(k_value)
        if k_int == 1:
            continue  # skip K=1 per requirement
        group = group.sort_values("N")
        x_values = group["N"]
        plt.plot(
            x_values,
            group["mean_acc_k"],
            color=style_by_k[int(k_value)]["color"],
            marker=style_by_k[int(k_value)]["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"K={k_int}",
        )

    plt.xlabel("N (multiple of K)")
    plt.ylabel("Final mean accuracy")
    plt.title("Countdown - Final mean accuracy vs N")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()

def plot_pass_vs_loops_for_N(df: pd.DataFrame, fixed_N: int = 16, out_path: Optional[str] = None) -> None:
    """Plot mean_pass_at_k vs loops for a fixed N, across different K values.

    - X-axis: loop
    - One line per k for the given N
    - Saves to out_path if provided, otherwise shows interactively
    """
    if df.empty:
        print("No data found to plot.")
        return

    sub = df[df["N"] == fixed_N].copy()
    if sub.empty:
        print(f"No rows found for N={fixed_N}.")
        return

    # Filter rows that actually have pass@k
    sub = sub[~sub["mean_pass_at_k"].isna()]
    if sub.empty:
        print("No mean_pass_at_k values to plot for the given N.")
        return

    style_by_k = {
        1: {"color": "C0", "marker": "o"},
        2: {"color": "C1", "marker": "^"},
        3: {"color": "C2", "marker": "s"},
        4: {"color": "C3", "marker": "D"},
    }

    plt.figure(figsize=(6, 4.5))
    plotted_ks = set()

    # Ensure k=1 is always included if present anywhere
    if 1 not in plotted_ks:
        k1_all = df[df["k"] == 1]
        if not k1_all.empty and "mean_pass_at_k" in k1_all.columns:
            k1_all = k1_all[~k1_all["mean_pass_at_k"].isna()]
            if not k1_all.empty:
                k1_candidates = k1_all.groupby("N")
                available_Ns = set(k1_candidates.groups.keys())
                preferred_N = 1 if 1 in available_Ns else None
                if preferred_N is not None:
                    k1_group = k1_candidates.get_group(preferred_N).sort_values("loop")
                    plt.plot(
                        k1_group["loop"],
                        k1_group["mean_pass_at_k"],
                        color=style_by_k[1]["color"],
                        marker=style_by_k[1]["marker"],
                        linestyle="-",
                        linewidth=2,
                        markersize=6,
                        label=f"K=1",
                    )

    for k_value, group in sub.groupby("k", sort=False):
        group = group.sort_values("loop")
        k_int = int(k_value)
        style = style_by_k.get(k_int, {"color": None, "marker": "o"})
        plt.plot(
            group["loop"],
            group["mean_pass_at_k"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=f"K={k_int}",
        )
        plotted_ks.add(int(k_value))

    plt.xlabel("Loop step")
    plt.ylabel("pass@k")
    plt.title(f"Countdown - pass@k vs loops (N={fixed_N})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Curves", fontsize=8)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Plot mean accuracies across loops for k/N configs.")
    ap.add_argument("--root", type=str, default="/pscratch/sd/m/mokshjn/eval/countdown/ref/Qwen3-4B-Instruct-2507/")
    ap.add_argument(
        "--dir-pattern",
        type=str,
        default=r"k_(?P<k>\d+)_N_(?P<N>\d+)",
    )
    ap.add_argument("--out", type=str, default="plot_base.png")
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    df = collect_results(args.root, args.dir_pattern)
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Saved CSV: {args.csv}")

    # plot_results(df, args.out)
    # plot_k_vs_loops_for_N(df, fixed_N=16, out_path=args.out)
    plot_final_acc_vs_N_by_K(df, final_loop=4, out_path="final_acc_vs_N_by_K.png")
    plot_pass_vs_loops_for_K(df, fixed_K=4, out_path="pass_vs_loops_for_K.png")
    plot_pass_minus_acc_vs_loops_for_N(df, fixed_N=16, out_path="pass_minus_acc_vs_loops_for_N.png")


if __name__ == "__main__":
    main()


