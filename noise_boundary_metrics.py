import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import noise_test
from clean_run_metrics import (
    COEFFICIENT_TOLERANCE,
    best_true_coefficient_match,
    coefficient_by_feature,
    framework_feature_normalizer,
    load_framework_module,
    normalize_result,
    normalize_target,
)


ROOT = Path(__file__).resolve().parent

DEFAULT_BOUNDARY_FILES = {
    "pysindy": ROOT / "results" / "pysindy_noisy" / "noise_manual_3_5_summary.csv",
    "deepmod": ROOT / "results" / "deepmod" / "noise_boundaries_3_5.csv",
}


def target_metrics(framework, dataset, result, target_index):
    result = normalize_result(result)
    raw_target = result["targets"][target_index]
    target_name = normalize_target(framework, raw_target, dataset)
    fitted_coefficients = coefficient_by_feature(
        result,
        target_index,
        dataset=dataset,
        feature_normalizer=framework_feature_normalizer(framework),
    )
    truth_match = best_true_coefficient_match(
        dataset,
        target_name,
        fitted_coefficients,
        tolerance=COEFFICIENT_TOLERANCE,
    )
    if truth_match is None:
        raise KeyError(f"No true coefficients configured for {dataset} / {target_name}")
    return {
        "target": target_name,
        "hd": truth_match["hamming"],
        "re": truth_match["relative_error_sum"],
    }


def selected_metrics(framework, dataset, target_name, result):
    result = normalize_result(result)
    target_rows = [
        target_metrics(framework, dataset, result, target_index)
        for target_index, _ in enumerate(result["targets"])
    ]
    if target_name in {"system", "__system__"}:
        return {
            "hd": sum(row["hd"] for row in target_rows),
            "re": sum(row["re"] for row in target_rows),
        }

    normalized_targets = [row["target"] for row in target_rows]
    target_index = normalized_targets.index(target_name)
    return target_rows[target_index]


def summarize(values):
    if not values:
        return {"mean": "", "std": ""}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def runner_args(args):
    return SimpleNamespace(
        device=args.device,
        solution_index=args.solution_index,
        epde_best_pareto=args.epde_best_pareto,
    )


def measure_boundary(framework, module, dataset, target_name, noise_level, runs, base_seed, args):
    hd_values = []
    correct_re_values = []
    correct_count = 0
    error_count = 0

    for run_index in range(runs):
        seed = noise_test.noise_seed(run_index, base_seed)
        try:
            result = noise_test.run_dataset_at_noise(
                dataset,
                noise_level,
                seed,
                framework=framework,
                module=module,
                args=runner_args(args),
            )
            metrics = selected_metrics(framework, dataset, target_name, result)
            hd_values.append(metrics["hd"])
            if metrics["hd"] == 0:
                correct_count += 1
                correct_re_values.append(metrics["re"])
        except Exception:
            error_count += 1
            hd_values.append(np.nan)

    valid_hd = [value for value in hd_values if not np.isnan(value)]
    hd_summary = summarize(valid_hd)
    re_summary = summarize(correct_re_values)
    return {
        "framework": framework,
        "dataset": dataset,
        "target": "system" if target_name == "__system__" else target_name,
        "noise_level": noise_level,
        "runs": runs,
        "valid_runs": len(valid_hd),
        "correct_count": correct_count,
        "re_runs": len(correct_re_values),
        "error_count": error_count,
        "hd_mean": hd_summary["mean"],
        "hd_std": hd_summary["std"],
        "re_mean": re_summary["mean"],
        "re_std": re_summary["std"],
    }


def boundary_from_row(row):
    if row.get("error"):
        return None
    if row.get("break_noise_level", "") != "":
        return row["dataset"], row["target"], float(row["break_noise_level"])
    if row.get("noise_level_3_5", "") != "":
        target = "__system__" if row["target"] == "system" else row["target"]
        return row["dataset"], target, float(row["noise_level_3_5"])
    if row.get("noise_level", "") != "":
        target = "__system__" if row["target"] == "system" else row["target"]
        return row["dataset"], target, float(row["noise_level"])
    return None


def load_boundaries(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    boundaries = []
    for row in rows:
        boundary = boundary_from_row(row)
        if boundary is not None:
            boundaries.append(boundary)
    return boundaries


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "dataset",
        "target",
        "noise_level",
        "runs",
        "valid_runs",
        "correct_count",
        "re_runs",
        "error_count",
        "hd_mean",
        "hd_std",
        "re_mean",
        "re_std",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows):
    print("\nNoise boundary HD/RE statistics:")
    for row in rows:
        re_text = (
            f"RE={row['re_mean']:.4g} +/- {row['re_std']:.4g}"
            if row["re_mean"] != "" else "RE=n/a"
        )
        print(
            f"  {row['framework']} / {row['dataset']} / {row['target']}: "
            f"noise={row['noise_level']}, "
            f"correct={row['correct_count']}/{row['runs']}, "
            f"HD={row['hd_mean']:.4g} +/- {row['hd_std']:.4g}, "
            f"{re_text}"
        )


def default_boundaries_csv(framework):
    if framework not in DEFAULT_BOUNDARY_FILES:
        raise ValueError(f"No default boundary CSV for {framework}. Pass --boundaries-csv.")
    return DEFAULT_BOUNDARY_FILES[framework]


def default_output(framework):
    return ROOT / "results" / framework / "noise_boundary_metrics.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=["pysindy", "deepmod", "epde", "discover", "edl"])
    parser.add_argument("--runs", type=int, default=noise_test.DEFAULT_RUNS)
    parser.add_argument("--boundaries-csv", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solution-index", type=int, default=0)
    parser.add_argument("--epde-best-pareto", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    module = load_framework_module(args.framework)
    boundaries_csv = Path(args.boundaries_csv) if args.boundaries_csv else default_boundaries_csv(args.framework)
    output_file = Path(args.output) if args.output else default_output(args.framework)
    rows = []

    for dataset, target_name, noise_level in load_boundaries(boundaries_csv):
        print(f"Measuring {args.framework} / {dataset} / {target_name} at noise={noise_level}")
        rows.append(
            measure_boundary(
                args.framework,
                module,
                dataset,
                target_name,
                noise_level,
                args.runs,
                noise_test.RANDOM_SEED,
                args,
            )
        )

    write_rows(rows, output_file)
    print_rows(rows)
    print(f"\nSaved metrics to {output_file}")


if __name__ == "__main__":
    main()
