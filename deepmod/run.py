import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path().absolute()))
sys.path.append(str(Path().absolute() / "deepmod" / "deepymod" / "src"))

from deepymod.model.sparse_estimators import PDEFIND, Threshold

from data.config import DEEPMOD_DATASETS, DEEPMOD_DEFAULTS, deepmod_params, sindy_params
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


RESULTS_DIR = Path("results/deepmod")
DATASETS = DEEPMOD_DATASETS


def save_combined_results(results):
    output_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as handle:
        json.dump([results], handle, indent=2)


def build_run_params(filename):
    if filename not in deepmod_params:
        raise KeyError(f"No DeepMoD params configured for {filename!r}")
    if filename not in sindy_params:
        raise KeyError(f"No shared library params configured for {filename!r}")

    params = dict(DEEPMOD_DEFAULTS)
    params.update(deepmod_params[filename])
    params["filename"] = filename
    params["sindy_config"] = sindy_params[filename]
    params["crop"] = params.get("crop", sindy_params[filename].get("crop", 0))
    return params


def direct_optimizer_config(params):
    optimizer_config = dict(params.get("direct_optimizer", {}))
    if "type" not in optimizer_config:
        raise KeyError("DeepMoD config must define direct_optimizer.type")
    optimizer_config.setdefault("coefficient_tol", 0.0)
    return optimizer_config


def build_deepmod_estimator(opt_config):
    estimator_type = opt_config.get("type", "threshold")
    if estimator_type == "pdefind":
        return PDEFIND(
            lam=opt_config.get("pdefind_lam", 1e-3),
            dtol=opt_config.get("pdefind_dtol", opt_config.get("threshold", 0.1)),
        )
    if estimator_type == "threshold":
        return Threshold(opt_config.get("threshold", 0.1))
    raise ValueError(f"Unknown DeepMoD sparse estimator: {estimator_type}")


def normalize_for_deepmod_estimator(features, target_values):
    feature_norms = np.linalg.norm(features, axis=0, keepdims=True)
    feature_norms[feature_norms == 0] = 1.0
    target_norm = np.linalg.norm(target_values)
    if target_norm == 0:
        target_norm = 1.0
    return features / feature_norms, target_values / target_norm


def refit_coefficients(features, target_values, active_mask, coefficient_tol=0.0):
    coefficients = np.zeros(features.shape[1], dtype=float)
    if not np.any(active_mask):
        return coefficients
    active_coefficients = np.linalg.lstsq(
        features[:, active_mask],
        target_values,
        rcond=None,
    )[0]
    coefficients[active_mask] = active_coefficients
    if coefficient_tol > 0:
        coefficients[np.abs(coefficients) < coefficient_tol] = 0.0
    return coefficients


def fit_deepmod_sparse_system(features, target_values, feature_names, target_name, opt_config):
    estimator = build_deepmod_estimator(opt_config)
    normed_features, normed_target = normalize_for_deepmod_estimator(features, target_values)
    mask_coefficients = np.atleast_1d(estimator.fit(normed_features, normed_target))
    active_mask = np.abs(mask_coefficients.reshape(-1)) > 0.0
    coefficients = refit_coefficients(
        features,
        target_values,
        active_mask,
        coefficient_tol=opt_config.get("coefficient_tol", 0.0),
    )

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


def run_deepmod(data, x, y, z, t, filename):
    """Run the DeepMoD benchmark wrapper on precomputed NumPy derivatives.

    DeepMoD is compared here through the shared benchmark derivative data and
    candidate libraries.
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
    optimizer_config = direct_optimizer_config(params)

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
        result = fit_deepmod_sparse_system(
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
            all_results.append(run_deepmod(data, x, y, z, t, dataset))
        except Exception as error:
            print(f"Error processing {dataset}: {error}")
        finally:
            print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    save_combined_results(all_results)
    print_library_summary(all_results)
    print("\nAll experiments completed!")
