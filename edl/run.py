import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="PDE_find")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDL_ROOT = Path(__file__).resolve().parent
EDL_SOURCE_ROOT = EDL_ROOT / "EDL"
RESULTS_DIR = Path("results/edl")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EDL_SOURCE_ROOT / "evaluation"))

from PDE_find import STRidge, TrainSTRidge

from data.config import EDL_DATASETS, EDL_DEFAULTS, edl_params, sindy_params
from utils.dataloader import load_data
from utils.derivatives import compute_derivative_bundle
from utils.sindy_library import (
    build_crop_slices,
    build_target_problem,
    configured_max_deriv_order,
    default_targets,
    default_variable_names,
    normalize_data_arrays,
)


DATASETS = EDL_DATASETS


def save_combined_results(results):
    output_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump([results], handle, indent=2)


def build_run_params(filename):
    if filename not in edl_params:
        raise KeyError(f"No EDL params configured for {filename!r}")
    if filename not in sindy_params:
        raise KeyError(f"No shared library params configured for {filename!r}")

    params = dict(EDL_DEFAULTS)
    params.update(edl_params[filename])
    params["filename"] = filename
    params["sindy_config"] = sindy_params[filename]
    params["crop"] = params.get("crop", sindy_params[filename].get("crop", 0))
    return params


def fit_edl_sparse_system(features, target_values, feature_names, target_name, opt_config):
    target_values = np.asarray(target_values, dtype=float).reshape(-1, 1)
    optimizer_type = opt_config.get("type", "STRidge")

    if optimizer_type == "STRidge":
        coefficients = STRidge(
            np.asarray(features, dtype=float),
            target_values,
            lam=opt_config.get("lam", 1e-5),
            maxit=opt_config.get("str_iters", 10),
            tol=opt_config.get("tol", 0.1),
            normalize=opt_config.get("normalize", 2),
        )
    elif optimizer_type == "TrainSTRidge":
        coefficients, _ = TrainSTRidge(
            np.asarray(features, dtype=float),
            target_values,
            lam=opt_config.get("lam", 1e-5),
            d_tol=opt_config.get("d_tol", 0.1),
            maxit=opt_config.get("maxit", 25),
            STR_iters=opt_config.get("str_iters", 10),
            l0_penalty=opt_config.get("l0_penalty"),
            normalize=opt_config.get("normalize", 2),
            split=opt_config.get("split", 0),
        )
    else:
        raise ValueError(f"Unknown EDL sparse optimizer: {optimizer_type}")

    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    coefficient_tol = opt_config.get("coefficient_tol", 0.0)
    if coefficient_tol > 0:
        coefficients[np.abs(coefficients) < coefficient_tol] = 0.0

    active_terms = [
        f"{coefficient:.4f} {feature}"
        for coefficient, feature in zip(coefficients, feature_names)
        if abs(coefficient) > 1e-12
    ]
    rhs = " + ".join(active_terms).replace("+ -", "- ") or "0"
    print(f"{target_name} = {rhs}")
    print()

    return {
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }


def run_edl(data, x, y, z, t, filename):
    """Run EDL's sparse regression backend on the shared benchmark library.

    The original EDL pipeline uses an LLM to generate candidate equations and
    then evaluates them. For reproducible benchmark runs without API keys, this
    wrapper compares the EDL STRidge backend on the same fixed feature matrices
    and target derivatives used by the other framework wrappers.
    """

    params = build_run_params(filename)
    sindy_config = params["sindy_config"]

    data_arrays = normalize_data_arrays(data)
    lib_config = sindy_config.get("library", {})
    variable_names = lib_config.get("variable_names", default_variable_names(data_arrays))
    targets = sindy_config.get("targets", default_targets(variable_names))
    derivatives = compute_derivative_bundle(
        data_arrays if len(data_arrays) > 1 else data_arrays[0],
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=variable_names,
        max_orders=configured_max_deriv_order(data_arrays[0].shape, sindy_config),
    )
    crop_slices = build_crop_slices(data_arrays[0].shape, params.get("crop", 0))
    optimizer_config = dict(params.get("optimizer", {}))

    results = []
    feature_names_by_target = []
    library_sizes = {}
    for target in targets:
        lhs_name, features, feature_names, target_values = build_target_problem(
            target,
            sindy_config,
            derivatives,
            crop_slices,
            data_arrays[0].shape,
            x,
            t,
        )
        result = fit_edl_sparse_system(
            features,
            target_values,
            feature_names,
            lhs_name,
            optimizer_config,
        )
        results.append(result)
        feature_names_by_target.append(feature_names)
        library_sizes[lhs_name] = len(feature_names)

    return {
        "dataset": filename.split(".")[0],
        "targets": [result["target"] for result in results],
        "coefficients": [result["coefficients"] for result in results],
        "features": feature_names_by_target,
        "library_sizes": library_sizes,
        "library_size": sum(library_sizes.values()),
    }


def print_library_summary(results):
    print("\nLibrary sizes:")
    for result in results:
        dataset = result["dataset"]
        for target, size in result.get("library_sizes", {}).items():
            print(f"  {dataset} / {target}: {size}")
        if len(result.get("targets", [])) > 1:
            print(f"  {dataset} / __system__: {result.get('library_size', '')}")


if __name__ == "__main__":
    selected_datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    all_results = []
    for dataset in selected_datasets:
        print(f"\n=== Processing {dataset} ===")
        start = time.perf_counter()
        try:
            data, x, y, z, t = load_data(dataset)
            all_results.append(run_edl(data, x, y, z, t, dataset))
        except Exception as error:
            print(f"Error processing {dataset}: {error}")
        finally:
            print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    save_combined_results(all_results)
    print_library_summary(all_results)
    print("\nAll experiments completed!")
