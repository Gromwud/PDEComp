import argparse
import contextlib
import csv
import io
import time
from pathlib import Path

import numpy as np

import run as sindy_run
from data.dataloader import load_data


RESULTS_DIR = Path("results/pysindy")
COEFFICIENT_TOLERANCE = 1e-12


TRUE_COEFFICIENTS = {
    "ode_data.npy": {
        "u_tt": {
            "u_t sin(2 t)": -1.0,
            "t": 1.5,
            "u": -4.0,
        },
    },
    "vdp_data.npy": {
        "u_tt": {
            "u_t": 0.2,
            "u^2 u_t": -0.2,
            "u": -1.0,
        },
    },
    "lotka_data.npy": {
        "x0_t": {
            "x0": 20.0,
            "x0 x1": -20.0,
        },
        "x1_t": {
            "x0 x1": 20.0,
            "x1": -20.0,
        },
    },
    "lorenz_data.npy": {
        "x0_t": {
            "x0": -10.0,
            "x1": 10.0,
        },
        "x1_t": {
            "x0": 28.0,
            "x0 x2": -1.0,
            "x1": -1.0,
        },
        "x2_t": {
            "x0 x1": 1.0,
            "x2": -8.0 / 3.0,
        },
    },
    "burgers_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": 0.1,
        },
    },
    "burgers_sln_100_data.csv": {
        "u_t": {
            "u u_x": -1.0,
        },
    },
    "ac_data.npy": {
        "u_t": {
            "u^3": -5.0,
            "u": 5.0,
        },
    },
    "kdv_data.mat": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
        },
    },
    "kdv_periodic_data.npy": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
            "sin(x) cos(t)": 1.0,
        },
    },
    "wave_data.csv": {
        "u_tt": {
            "u_xx": 0.04,
        },
    },
    "pde_compound_data.npy": {
        "u_t": {
            "d_x(u u_x)": 1.0,
        },
    },
    "pde_divide_data.npy": {
        "u_t": {
            "u_xx": 0.25,
            "(1/x) u_x": -1.0,
        },
    },
    "ks_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": -1.0,
            "u_xxxx": -1.0,
        },
    },
    "ns_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "v u_y": -1.0,
            "p_x": -1.0,
            "(u_xx + u_yy)": 0.01,
        },
        "v_t": {
            "u v_x": -1.0,
            "v v_y": -1.0,
            "p_y": -1.0,
            "(v_xx + v_yy)": 0.01,
        },
        "u_x": {
            "v_y": -1.0,
        },
    },
    "ODE_simple_discovery": {
        "u_t": {
            "cos(t)": 1.0,
            "sin(t)": -1.3,
        },
    },
}


def coefficient_by_feature(result, target_index):
    """Return fitted coefficients keyed by feature name for one target."""

    coefficients = np.asarray(result["coefficients"][target_index], dtype=float)
    features = result["features"][target_index]
    return {
        feature: float(coefficient)
        for feature, coefficient in zip(features, coefficients)
    }


def active_features(coefficient_map, tolerance=COEFFICIENT_TOLERANCE):
    """Return fitted nonzero feature names."""

    return {
        feature
        for feature, coefficient in coefficient_map.items()
        if abs(coefficient) > tolerance
    }


def relative_error_sum(fitted_coefficients, true_coefficients):
    """Sum |fitted - true| / |true| over true nonzero coefficients."""

    total = 0.0
    for feature, true_value in true_coefficients.items():
        fitted_value = fitted_coefficients.get(feature, 0.0)
        total += abs(fitted_value - true_value) / abs(true_value)
    return total


def format_feature_list(features):
    return "; ".join(sorted(features))


def summarize_target(dataset, result, target_index, runtime_seconds):
    """Build one CSV row with clean-run metrics for a target equation."""

    target = result["targets"][target_index]
    features = result["features"][target_index]
    fitted_coefficients = coefficient_by_feature(result, target_index)
    true_coefficients = TRUE_COEFFICIENTS.get(dataset, {}).get(target)
    active = active_features(fitted_coefficients)

    if true_coefficients is None:
        expected = set()
        missing = set()
        extra = active
        error_sum = ""
        truth_defined = False
    else:
        expected = set(true_coefficients)
        missing = expected - active
        extra = active - expected
        error_sum = relative_error_sum(fitted_coefficients, true_coefficients)
        truth_defined = True

    return {
        "dataset": dataset,
        "target": target,
        "runtime_seconds": runtime_seconds,
        "library_size": len(features),
        "truth_defined": truth_defined,
        "true_terms_count": len(expected),
        "active_terms_count": len(active),
        "relative_error_sum": error_sum,
        "missing_terms": format_feature_list(missing),
        "extra_terms": format_feature_list(extra),
        "expected_terms": format_feature_list(expected),
        "active_terms": format_feature_list(active),
    }


def summarize_dataset(dataset, quiet=True):
    """Run PySINDy on clean data once and return target-level metric rows."""

    data, x, y, z, t = load_data(dataset)
    start = time.perf_counter()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = sindy_run.run_sindy(data, x, y, z, t, dataset)
    else:
        result = sindy_run.run_sindy(data, x, y, z, t, dataset)
    runtime_seconds = time.perf_counter() - start

    rows = [
        summarize_target(dataset, result, target_index, runtime_seconds)
        for target_index, _ in enumerate(result["targets"])
    ]
    if len(rows) > 1:
        rows.append(summarize_system(dataset, rows, runtime_seconds))
    return rows


def summarize_system(dataset, target_rows, runtime_seconds):
    """Add an aggregate row for multi-equation systems."""

    error_values = [
        row["relative_error_sum"]
        for row in target_rows
        if row["relative_error_sum"] != ""
    ]
    truth_defined = len(error_values) == len(target_rows)
    return {
        "dataset": dataset,
        "target": "__system__",
        "runtime_seconds": runtime_seconds,
        "library_size": sum(row["library_size"] for row in target_rows),
        "truth_defined": truth_defined,
        "true_terms_count": sum(row["true_terms_count"] for row in target_rows),
        "active_terms_count": sum(row["active_terms_count"] for row in target_rows),
        "relative_error_sum": sum(error_values) if truth_defined else "",
        "missing_terms": " | ".join(row["missing_terms"] for row in target_rows),
        "extra_terms": " | ".join(row["extra_terms"] for row in target_rows),
        "expected_terms": "",
        "active_terms": "",
    }


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "target",
        "runtime_seconds",
        "library_size",
        "truth_defined",
        "true_terms_count",
        "active_terms_count",
        "relative_error_sum",
        "missing_terms",
        "extra_terms",
        "expected_terms",
        "active_terms",
        "error",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=sindy_run.DATASETS)
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "clean_run_metrics.csv"),
    )
    parser.add_argument("--show-equations", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_rows = []

    for dataset in args.datasets:
        print(f"\n=== Measuring {dataset} ===")
        try:
            rows = summarize_dataset(dataset, quiet=not args.show_equations)
            all_rows.extend(rows)
            for row in rows:
                print(
                    f"  {row['target']}: time={row['runtime_seconds']:.4f}s, "
                    f"library={row['library_size']}, "
                    f"rel_error_sum={row['relative_error_sum']}"
                )
        except Exception as error:
            all_rows.append({
                "dataset": dataset,
                "target": "",
                "runtime_seconds": "",
                "library_size": "",
                "truth_defined": False,
                "true_terms_count": "",
                "active_terms_count": "",
                "relative_error_sum": "",
                "missing_terms": "",
                "extra_terms": "",
                "expected_terms": "",
                "active_terms": "",
                "error": str(error),
            })
            print(f"  error: {error}")

    output_file = Path(args.output)
    write_rows(all_rows, output_file)
    print(f"\nSaved clean-run metrics to {output_file}")
