import argparse
import contextlib
import csv
import io
from pathlib import Path

import numpy as np

from clean_run_metrics import (
    COEFFICIENT_TOLERANCE,
    DEFAULT_DATASETS,
    framework_feature_normalizer,
    load_framework_module,
    normalize_result,
    normalize_target,
    select_best_epde_candidate,
)
from utils.dataloader import load_data


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = 30
RANDOM_SEED = 42
DEFAULT_NOISE_SCALE = 0.01
DEFAULT_SUCCESS_MIN = 3
DEFAULT_SUCCESS_MAX = 5

DEFAULT_LEVELS = [
    0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04,
    0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2,
    3, 5, 7.5, 10, 15, 20, 25,
]
EPDE_DEFAULT_LEVELS = [0, 25, 50, 75]


def noise_seed(run_index, base_seed=RANDOM_SEED):
    return base_seed + run_index


def set_random_seed(seed):
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise(data, noise_level, scale=DEFAULT_NOISE_SCALE):
    return data + noise_level * scale * np.std(data) * np.random.normal(size=data.shape)


def add_dataset_noise(data, noise_level):
    if isinstance(data, list):
        return [add_noise(values, noise_level) for values in data]
    return add_noise(data, noise_level)


def active_terms(result, target_index, dataset=None, framework="pysindy", tolerance=COEFFICIENT_TOLERANCE):
    result = normalize_result(result)
    coefficients = np.asarray(result["coefficients"][target_index], dtype=float)
    features = result["features"][target_index]
    normalizer = framework_feature_normalizer(framework)
    return {
        normalizer(feature, dataset)
        for coefficient, feature in zip(coefficients, features)
        if abs(coefficient) > tolerance
    }


def result_target(result, target_index, dataset=None, framework="pysindy"):
    result = normalize_result(result)
    return normalize_target(framework, result["targets"][target_index], dataset)


def run_framework_on_data(framework, module, data, x, y, z, t, dataset, args=None):
    if framework == "pysindy":
        return module.run_sindy(data, x, y, z, t, dataset)
    if framework == "deepmod":
        return module.run_deepmod(data, x, y, z, t, dataset)
    if framework == "epde":
        device = getattr(args, "device", "cpu")
        solution_index = getattr(args, "solution_index", 0)
        return_all = getattr(args, "epde_best_pareto", False)
        return module.run_epde(
            data,
            x,
            y,
            z,
            t,
            dataset,
            device=device,
            solution_index=solution_index,
            only_print=False,
            visualize=False,
            return_all=return_all,
        )
    raise ValueError(f"Unknown framework: {framework}")


def run_dataset_at_noise(dataset, noise_level, seed, framework="pysindy", module=None, args=None):
    set_random_seed(seed)
    module = module or load_framework_module(framework)
    data, x, y, z, t = load_data(dataset)
    noised_data = add_dataset_noise(data, noise_level)

    with contextlib.redirect_stdout(io.StringIO()):
        result = run_framework_on_data(framework, module, noised_data, x, y, z, t, dataset, args=args)

    if framework == "epde" and getattr(args, "epde_best_pareto", False):
        result = select_best_epde_candidate(dataset, result)

    return normalize_result(result)


def summarize_dataset(framework, module, dataset, levels, runs, base_seed, args):
    rows = []
    clean_result = run_dataset_at_noise(dataset, 0, base_seed, framework=framework, module=module, args=args)
    clean_targets = [
        result_target(clean_result, target_index, dataset=dataset, framework=framework)
        for target_index, _ in enumerate(clean_result["targets"])
    ]
    clean_terms_by_target = [
        active_terms(clean_result, target_index, dataset=dataset, framework=framework)
        for target_index, _ in enumerate(clean_result["targets"])
    ]
    is_system = len(clean_targets) > 1

    for noise_level in levels:
        try:
            target_stats = {
                target_name: {
                    "clean_terms": set(clean_terms_by_target[target_index]),
                    "success_count": 0,
                    "successful_runs": [],
                    "successful_seeds": [],
                }
                for target_index, target_name in enumerate(clean_targets)
            }
            system_success_count = 0
            system_successful_runs = []
            system_successful_seeds = []

            for run_index in range(runs):
                seed = noise_seed(run_index, base_seed)
                result = clean_result if noise_level == 0 else run_dataset_at_noise(
                    dataset,
                    noise_level,
                    seed,
                    framework=framework,
                    module=module,
                    args=args,
                )
                result_targets = [
                    result_target(result, index, dataset=dataset, framework=framework)
                    for index, _ in enumerate(result["targets"])
                ]
                system_success = result_targets == clean_targets

                for target_index, target_name in enumerate(clean_targets):
                    target_success = False
                    if target_index < len(result_targets) and result_targets[target_index] == target_name:
                        noisy_terms = active_terms(result, target_index, dataset=dataset, framework=framework)
                        target_success = noisy_terms == target_stats[target_name]["clean_terms"]

                    if target_success:
                        target_stats[target_name]["success_count"] += 1
                        target_stats[target_name]["successful_runs"].append(run_index)
                        target_stats[target_name]["successful_seeds"].append(seed)
                    else:
                        system_success = False

                if system_success:
                    system_success_count += 1
                    system_successful_runs.append(run_index)
                    system_successful_seeds.append(seed)

            for target_name, stats in target_stats.items():
                success_count = stats["success_count"]
                rows.append({
                    "framework": framework,
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "target": target_name,
                    "runs": runs,
                    "success_count": success_count,
                    "has_success": success_count > 0,
                    "in_target_band": args.success_min <= success_count <= args.success_max,
                    "clean_terms_count": len(stats["clean_terms"]),
                    "successful_runs": "; ".join(map(str, stats["successful_runs"])),
                    "successful_seeds": "; ".join(map(str, stats["successful_seeds"])),
                    "error": "",
                })

            if is_system:
                rows.append({
                    "framework": framework,
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "target": "__system__",
                    "runs": runs,
                    "success_count": system_success_count,
                    "has_success": system_success_count > 0,
                    "in_target_band": args.success_min <= system_success_count <= args.success_max,
                    "clean_terms_count": sum(len(stats["clean_terms"]) for stats in target_stats.values()),
                    "successful_runs": "; ".join(map(str, system_successful_runs)),
                    "successful_seeds": "; ".join(map(str, system_successful_seeds)),
                    "error": "",
                })
        except Exception as error:
            rows.append({
                "framework": framework,
                "dataset": dataset,
                "noise_level": noise_level,
                "target": "",
                "runs": runs,
                "success_count": 0,
                "has_success": False,
                "in_target_band": False,
                "clean_terms_count": "",
                "successful_runs": "",
                "successful_seeds": "",
                "error": str(error),
            })
    return rows


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "dataset",
        "noise_level",
        "target",
        "runs",
        "success_count",
        "has_success",
        "in_target_band",
        "clean_terms_count",
        "successful_runs",
        "successful_seeds",
        "error",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_target_band_levels(rows, success_min=DEFAULT_SUCCESS_MIN, success_max=DEFAULT_SUCCESS_MAX):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["framework"], row["dataset"], row["target"]), []).append(row)

    print(f"\nNoise levels with {success_min}-{success_max} successful runs:")
    for (framework, dataset, target), target_rows in grouped.items():
        has_system_row = any(
            row["framework"] == framework
            and row["dataset"] == dataset
            and row["target"] == "__system__"
            for row in rows
        )
        if has_system_row and target != "__system__":
            continue
        matching = [
            row for row in target_rows
            if not row.get("error")
            and success_min <= int(row["success_count"]) <= success_max
        ]
        if matching:
            best = matching[-1]
            print(
                f"  {framework} / {dataset} / {target}: {best['noise_level']} "
                f"({best['success_count']}/{best['runs']})"
            )
        else:
            print(f"  {framework} / {dataset} / {target}: none")


def default_levels(framework):
    return EPDE_DEFAULT_LEVELS if framework == "epde" else DEFAULT_LEVELS


def default_output(framework):
    if framework == "pysindy":
        return ROOT / "results" / "pysindy_noisy" / "noise_success_summary.csv"
    if framework == "epde":
        return ROOT / "results" / "epde_noisy" / "noise_success_summary.csv"
    return ROOT / "results" / framework / "noise_success_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Universal noisy-run benchmark for PDE discovery frameworks.")
    parser.add_argument("framework", choices=["pysindy", "deepmod", "epde"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--levels", nargs="*", type=float, default=None)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--success-min", type=int, default=DEFAULT_SUCCESS_MIN)
    parser.add_argument("--success-max", type=int, default=DEFAULT_SUCCESS_MAX)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solution-index", type=int, default=0)
    parser.add_argument("--epde-best-pareto", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    module = load_framework_module(args.framework)
    datasets = args.datasets or getattr(module, "DATASETS", DEFAULT_DATASETS)
    levels = args.levels if args.levels is not None else default_levels(args.framework)
    output_file = Path(args.output) if args.output else default_output(args.framework)
    all_rows = []

    if not datasets:
        print("No datasets selected. Pass one or more names via --datasets.")

    for dataset in datasets:
        print(f"\n=== Sweeping {args.framework} / {dataset} ===")
        all_rows.extend(
            summarize_dataset(
                args.framework,
                module,
                dataset,
                levels,
                args.runs,
                RANDOM_SEED,
                args,
            )
        )

    write_rows(all_rows, output_file)
    print_target_band_levels(all_rows, args.success_min, args.success_max)
    print(f"\nSaved sweep summary to {output_file}")


if __name__ == "__main__":
    main()
