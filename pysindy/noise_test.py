import argparse
import contextlib
import csv
import io
from pathlib import Path

import numpy as np

import run as sindy_run
from data.dataloader import load_data


RESULTS_DIR = Path("results/pysindy_noisy")
DEFAULT_LEVELS = [0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 15, 20, 21, 22, 30]
DEFAULT_RUNS = 30
RANDOM_SEED = 42
DEFAULT_NOISE_SCALE = 0.01


def noise_seed(run_index, base_seed=RANDOM_SEED):
    """Return the seed shared by all methods for the same run index."""

    return base_seed + run_index


def add_noise(data, noise_level, scale=DEFAULT_NOISE_SCALE):
    """Add Gaussian noise proportional to data std."""

    return noise_level * scale * np.std(data) * np.random.normal(size=data.shape) + data


def add_dataset_noise(data, noise_level):
    """Apply noise independently to every data variable."""

    if isinstance(data, list):
        return [
            add_noise(values, noise_level)
            for values in data
        ]
    return add_noise(data, noise_level)


def active_terms(result, target_index, tolerance=1e-12):
    """Return active feature names for one fitted target."""

    coefficients = np.asarray(result["coefficients"][target_index], dtype=float)
    features = result["features"][target_index]
    return [
        feature
        for coefficient, feature in zip(coefficients, features)
        if abs(coefficient) > tolerance
    ]


def run_dataset_at_noise(dataset, noise_level, seed):
    """Run one dataset with a fixed noise level and seed."""

    np.random.seed(seed)
    data, x, y, z, t = load_data(dataset)
    noised_data = add_dataset_noise(data, noise_level)

    with contextlib.redirect_stdout(io.StringIO()):
        result = sindy_run.run_sindy(noised_data, x, y, z, t, dataset)

    return result


def summarize_dataset(dataset, levels, runs, base_seed):
    """Count successful noisy runs for every target and noise level."""

    rows = []
    clean_result = run_dataset_at_noise(dataset, 0, base_seed)
    clean_terms_by_target = [
        active_terms(clean_result, target_index)
        for target_index, _ in enumerate(clean_result["targets"])
    ]
    is_system = len(clean_result["targets"]) > 1

    for noise_level in levels:
        try:
            target_stats = {
                target_name: {
                    "clean_terms": set(clean_terms_by_target[target_index]),
                    "success_count": 0,
                    "successful_runs": [],
                    "successful_seeds": [],
                }
                for target_index, target_name in enumerate(clean_result["targets"])
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
                )
                system_success = True
                for target_index, target_name in enumerate(result["targets"]):
                    clean_terms = target_stats[target_name]["clean_terms"]
                    noisy_terms = set(active_terms(result, target_index))
                    if noisy_terms == clean_terms:
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
                rows.append({
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "target": target_name,
                    "runs": runs,
                    "success_count": stats["success_count"],
                    "has_success": stats["success_count"] > 0,
                    "clean_terms_count": len(stats["clean_terms"]),
                    "successful_runs": "; ".join(map(str, stats["successful_runs"])),
                    "successful_seeds": "; ".join(map(str, stats["successful_seeds"])),
                })
            if is_system:
                rows.append({
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "target": "__system__",
                    "runs": runs,
                    "success_count": system_success_count,
                    "has_success": system_success_count > 0,
                    "clean_terms_count": sum(
                        len(stats["clean_terms"])
                        for stats in target_stats.values()
                    ),
                    "successful_runs": "; ".join(map(str, system_successful_runs)),
                    "successful_seeds": "; ".join(map(str, system_successful_seeds)),
                })
        except Exception as error:
            rows.append({
                "dataset": dataset,
                "noise_level": noise_level,
                "target": "",
                "runs": runs,
                "success_count": 0,
                "has_success": False,
                "clean_terms_count": "",
                "successful_runs": "",
                "successful_seeds": "",
                "error": str(error),
            })
    return rows


def write_rows(rows, output_file):
    """Write the sweep summary as CSV."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "noise_level",
        "target",
        "runs",
        "success_count",
        "has_success",
        "clean_terms_count",
        "successful_runs",
        "successful_seeds",
        "error",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_max_success_levels(rows):
    """Print max noise level with at least one successful run."""

    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["target"]), []).append(row)

    print("\nMax noise level with at least one successful run:")
    for (dataset, target), target_rows in grouped.items():
        has_system_row = any(
            row["dataset"] == dataset and row["target"] == "__system__"
            for row in rows
        )
        if has_system_row and target != "__system__":
            continue
        successful = [
            row for row in target_rows
            if not row.get("error") and row.get("has_success")
        ]
        if successful:
            best = successful[-1]
            print(
                f"  {dataset} / {target}: {best['noise_level']} "
                f"({best['success_count']}/{best['runs']})"
            )
        else:
            print(f"  {dataset} / {target}: none")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=sindy_run.DATASETS)
    parser.add_argument("--levels", nargs="*", type=float, default=DEFAULT_LEVELS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "noise_success_summary.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_rows = []

    for dataset_index, dataset in enumerate(args.datasets):
        print(f"\n=== Sweeping {dataset} ===")
        rows = summarize_dataset(
            dataset,
            args.levels,
            args.runs,
            RANDOM_SEED,
        )
        all_rows.extend(rows)

    output_file = Path(args.output)
    write_rows(all_rows, output_file)
    print_max_success_levels(all_rows)
    print(f"\nSaved sweep summary to {output_file}")
