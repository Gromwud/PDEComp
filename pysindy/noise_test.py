import argparse
import contextlib
import csv
import io
from pathlib import Path

import numpy as np

import noise_test
import run as sindy_run
from data.dataloader import load_data



RESULTS_DIR = Path("results/pysindy_noisy")
DEFAULT_LEVELS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
RANDOM_SEED = 42
DEFAULT_NOISE_SCALE = 0.01


def add_noise(data, noise_level, scale=DEFAULT_NOISE_SCALE):
    """Add Gaussian noise proportional to data std."""

    return noise_level * scale * np.std(data) * np.random.normal(size=data.shape) + data


def add_dataset_noise(data, filename, noise_level):
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
    noised_data = add_dataset_noise(data, dataset, noise_level)

    with contextlib.redirect_stdout(io.StringIO()):
        result = sindy_run.run_sindy(noised_data, x, y, z, t, dataset)

    return result


def summarize_dataset(dataset, levels, base_seed):
    """Compare noisy active terms with the clean active terms."""

    rows = []
    clean_result = run_dataset_at_noise(dataset, 0, base_seed)
    clean_terms_by_target = [
        active_terms(clean_result, target_index)
        for target_index, _ in enumerate(clean_result["targets"])
    ]

    for level_index, noise_level in enumerate(levels):
        try:
            result = clean_result if noise_level == 0 else run_dataset_at_noise(
                dataset,
                noise_level,
                base_seed + level_index,
            )
            for target_index, target_name in enumerate(result["targets"]):
                clean_terms = set(clean_terms_by_target[target_index])
                noisy_terms = set(active_terms(result, target_index))
                extra_terms = sorted(noisy_terms - clean_terms)
                missing_terms = sorted(clean_terms - noisy_terms)
                rows.append({
                    "dataset": dataset,
                    "noise_level": noise_level,
                    "target": target_name,
                    "clean_terms_count": len(clean_terms),
                    "noisy_terms_count": len(noisy_terms),
                    "extra_terms_count": len(extra_terms),
                    "missing_terms_count": len(missing_terms),
                    "same_terms": not extra_terms and not missing_terms,
                    "extra_terms": "; ".join(extra_terms),
                    "missing_terms": "; ".join(missing_terms),
                })
        except Exception as error:
            rows.append({
                "dataset": dataset,
                "noise_level": noise_level,
                "target": "",
                "clean_terms_count": "",
                "noisy_terms_count": "",
                "extra_terms_count": "",
                "missing_terms_count": "",
                "same_terms": False,
                "extra_terms": "",
                "missing_terms": "",
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
        "clean_terms_count",
        "noisy_terms_count",
        "extra_terms_count",
        "missing_terms_count",
        "same_terms",
        "extra_terms",
        "missing_terms",
        "error",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_breakpoints(rows):
    """Print the first noise level where active terms change."""

    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["target"]), []).append(row)

    print("\nFirst changed noise level:")
    for (dataset, target), target_rows in grouped.items():
        changed = [
            row for row in target_rows
            if row.get("error") or not row["same_terms"]
        ]
        if changed:
            first = changed[0]
            print(
                f"  {dataset} / {target}: {first['noise_level']} "
                f"(extra={first['extra_terms_count']}, missing={first['missing_terms_count']})"
            )
        else:
            print(f"  {dataset} / {target}: unchanged")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=sindy_run.DATASETS)
    parser.add_argument("--levels", nargs="*", type=float, default=DEFAULT_LEVELS)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "noise_sweep_summary.csv"),
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
            RANDOM_SEED + dataset_index * 1000,
        )
        all_rows.extend(rows)

    output_file = Path(args.output)
    write_rows(all_rows, output_file)
    print_breakpoints(all_rows)
    print(f"\nSaved sweep summary to {output_file}")
