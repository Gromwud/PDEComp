import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISCOVER_ROOT = Path(__file__).resolve().parent
DISCOVER_FORK_ROOT = DISCOVER_ROOT / "discover"
DSO_ROOT = DISCOVER_FORK_ROOT / "dso"
RESULTS_DIR = Path("results/discover")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DSO_ROOT))

from data.config import DISCOVER_DATASETS, DISCOVER_DEFAULTS, discover_params, sindy_params
from utils.derivatives import compute_derivative_bundle, get_derivative
from utils.dataloader import load_data
from utils.sindy_library import (
    build_crop_slices,
    build_target_problem,
    configured_max_deriv_order,
    default_targets,
)


DATASETS = DISCOVER_DATASETS


def save_combined_results(results):
    output_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump([results], handle, indent=2)


def build_run_params(filename):
    if filename not in discover_params:
        raise KeyError(f"No DISCOVER params configured for {filename!r}")
    params = dict(DISCOVER_DEFAULTS)
    params.update(discover_params[filename])
    params["sindy_config"] = sindy_params[filename]
    return params


def discover_config_path(params):
    return DSO_ROOT / "dso" / "config" / "MODE1" / params["base_config"]


def scalar_values(data, filename):
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"DISCOVER wrapper supports scalar equations only, got a system in {filename}")
        data = data[0]
    return np.asarray(data, dtype=float)


def build_external_problem(data, x, y, z, t, filename, params):
    values = scalar_values(data, filename)
    if values.ndim not in (1, 2):
        raise ValueError(f"DISCOVER wrapper supports scalar ODE and 1D PDE data, got {filename}")
    if values.ndim == 2 and x is None:
        raise ValueError(f"DISCOVER 1D PDE data requires an x grid for {filename}")

    sindy_config = params["sindy_config"]
    bundle = compute_derivative_bundle(
        values,
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=["u"],
        max_orders=configured_max_deriv_order(values.shape, sindy_config),
    )
    target = sindy_config.get("targets", default_targets(["u"]))[0]
    crop_slices = build_crop_slices(values.shape, sindy_config.get("crop", 0))
    target_name, features, feature_names, target_values = build_target_problem(
        target,
        sindy_config,
        bundle,
        crop_slices,
        values.shape,
        x,
        t,
    )

    if values.ndim == 1:
        u = values.reshape(-1, 1)
        discover_x = [np.zeros((values.size, 1), dtype=float)]
    else:
        u = values.T
        discover_x = [np.asarray(x, dtype=float).reshape(-1, 1)]

    derivatives = {}
    if values.ndim == 2:
        for order in range(1, 5):
            try:
                derivatives[order] = get_derivative(bundle, "u", "x", order).T
            except KeyError:
                pass

    return {
        "u": [u],
        "x": discover_x,
        "t": np.asarray(t, dtype=float).reshape(-1, 1),
        "ut": target_values.reshape(-1, 1),
        "features": features,
        "feature_names": feature_names,
        "derivatives": derivatives,
        "sym_true": params.get("sym_true"),
        "target_name": target_name,
        "n_input_var": 1,
        "n_state_var": 1,
    }


def discover_overrides(filename, params):
    return {
        "experiment": {
            "logdir": str(RESULTS_DIR),
            "seed": params.get("seed", 0),
        },
        "task": {
            "task_type": "pde",
            "dataset": filename,
            "function_set": params["function_set"],
            "metric": "pde_reward",
            "metric_params": [params.get("metric_param", 0.01)],
            "threshold": params.get("threshold", 5e-4),
            "protected": params.get("protected", False),
            "decision_tree_threshold_set": [],
            "eq_num": 1,
            "spatial_error": params.get("spatial_error", False),
        },
        "training": {
            "n_samples": params["n_samples"],
            "batch_size": params["batch_size"],
            "epsilon": params["epsilon"],
            "n_cores_batch": params["n_cores_batch"],
            "early_stopping": params["early_stopping"],
            "verbose": params.get("verbose", True),
        },
        "controller": {
            "attention": False,
        },
        "prior": {
            "length": {"min_": 1, "max_": params.get("max_length", 15), "on": True},
            "repeat": {"tokens": "add", "min_": None, "max_": params.get("max_add_count", 8), "on": True},
            "inverse": {"on": False},
            "trig": {"on": False},
            "diff_left": {"on": False},
            "diff_right": {"on": False},
            "diff_descedent": {"on": False},
            "soft_length": {"loc": 5, "scale": 3, "on": True},
        },
        "gp_meld": {
            "run_gp_meld": False,
        },
        "gp_agg": {
            "run_gp_agg": False,
        },
        "pinn": {
            "use_pinn": False,
        },
        "parameterized": {
            "on": False,
        },
    }


def term_to_feature(term, feature_names=None):
    text = repr(term)
    if feature_names and text.startswith("theta_"):
        try:
            index = int(text.split("_", 1)[1])
            return feature_names[index]
        except (ValueError, IndexError):
            return text

    replacements = {
        "u1": "u",
        "diff(u1,x1)": "u_x",
        "diff2(u1,x1)": "u_xx",
        "diff3(u1,x1)": "u_xxx",
        "diff4(u1,x1)": "u_xxxx",
        "n2(u1)": "u^2",
        "n3(u1)": "u^3",
        "mul(u1,diff(u1,x1))": "u u_x",
        "mul(diff(u1,x1),u1)": "u u_x",
        "mul(u,diff(u,x1))": "u u_x",
        "mul(u1,diff2(u1,x1))": "u u_xx",
        "mul(diff2(u1,x1),u1)": "u u_xx",
        "mul(diff(u1,x1),diff(u1,x1))": "u_x^2",
        "div(diff(u1,x1),x1)": "(1/x) u_x",
        "mul(sin(x1),cos(x2))": "sin(x) cos(t)",
    }
    return replacements.get(text, text)


def extract_result(train_result, filename, target_name, elapsed_time, params):
    from dso.task.pde import data_load

    library_data = data_load.get_external_library(filename) or {}
    feature_names = library_data.get("feature_names")

    program = train_result["program"]
    terms = getattr(program.STRidge, "terms", [])
    coefficients = [float(value) for value in np.asarray(program.w, dtype=float).reshape(-1)]
    selected_features = [term_to_feature(term, feature_names) for term in terms]

    if feature_names:
        features = list(feature_names)
        coefficient_by_feature = dict(zip(selected_features, coefficients))
        coefficients = [coefficient_by_feature.get(feature_name, 0.0) for feature_name in features]
    else:
        features = selected_features
        if len(coefficients) > len(features):
            features.append("__constant__")

    coefficient_tol = (
        params
        .get("sindy_config", {})
        .get("optimizer", {})
        .get("coefficient_tol", 1e-12)
    )
    coefficients = [
        0.0 if abs(coefficient) < coefficient_tol else coefficient
        for coefficient in coefficients
    ]

    return {
        "dataset": filename.split(".")[0],
        "targets": [target_name],
        "features": [features],
        "coefficients": [coefficients[: len(features)]],
        "active_terms": [[feature for feature, coefficient in zip(features, coefficients) if abs(coefficient) > 1e-12]],
        "library_sizes": {target_name: len(features)},
        "library_size": len(features),
        "model": train_result.get("expression", ""),
        "equation_texts": [train_result.get("expression", "")],
        "time": elapsed_time,
        "discover_reward": train_result.get("r", ""),
        "discover_nmse": train_result.get("nmse_test", ""),
    }


def run_discover(data, x, y, z, t, filename, only_print=True):
    params = build_run_params(filename)

    try:
        from dso import DeepSymbolicOptimizer_PDE
        from dso.task.pde import data_load
    except Exception as error:
        raise RuntimeError(
            "DISCOVER is expected in discover/discover/dso, but its TensorFlow 1.x "
            "stack is not importable in the current environment."
        ) from error

    problem = build_external_problem(data, x, y, z, t, filename, params)
    data_load.set_external_pde_problem(filename, **problem)
    start = time.perf_counter()
    try:
        model = DeepSymbolicOptimizer_PDE(
            str(discover_config_path(params)),
            pde_config=discover_overrides(filename, params),
        )
        raw_result = model.train()
        train_result = raw_result[0] if isinstance(raw_result, list) else raw_result
        elapsed_time = time.perf_counter() - start
        result = extract_result(train_result, filename, params["target"], elapsed_time, params)
    finally:
        data_load.clear_external_pde_problem(filename)

    if only_print:
        print(result["model"])
    return result


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            all_results.append(run_discover(data, x, y, z, t, dataset))
        except Exception as error:
            print(f"Error processing {dataset}: {error}")

    if all_results:
        save_combined_results(all_results)
    print("\nAll experiments completed!")
