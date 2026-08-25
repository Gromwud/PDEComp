import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pysindy as ps

sys.path.append(str(Path().absolute()))

from data.config import sindy_params
from utils import sindy_library
from utils.dataloader import load_data
from utils.derivatives import compute_derivative_bundle


RESULTS_DIR = Path("results/pysindy")
DATASETS = [
    "ode_data.npy",
    "vdp_data.npy",
    "lorenz_data.npy",
    "lotka_data.npy",
    "burgers_data.mat",
    "ac_data.npy",
    "kdv_data.mat",
    "kdv_periodic_data.npy",
    "wave_data.csv",
    "pde_divide_data.npy",
    "pde_compound_data.npy",
    "ns_data.mat",
    "ks_data.mat",
    "burgers_sln_100_data.csv",
    "ODE_simple_discovery",
]


def save_combined_results(results):
    """Save all dataset results into one timestamped JSON file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as handle:
        json.dump([results], handle, indent=2)


def build_optimizer(opt_config):
    """Create the sparse optimizer described by the dataset config."""

    opt_type = opt_config.get("type", "STLSQ")

    if opt_type == "STLSQ":
        return ps.STLSQ(
            threshold=opt_config.get("threshold", 0.1),
            alpha=opt_config.get("alpha", 0.05),
            normalize_columns=opt_config.get("normalize_columns", False),
        )

    if opt_type == "SR3":
        return ps.SR3(
            threshold=opt_config.get("threshold", 0.1),
            max_iter=opt_config.get("max_iter", 30),
            tol=opt_config.get("tol", 1e-5),
            normalize_columns=opt_config.get("normalize_columns", False),
            thresholder=opt_config.get("thresholder", opt_config.get("regularizer", "L0")),
            nu=opt_config.get("nu", 1.0),
        )

    if opt_type == "FROLS":
        return ps.FROLS(
            max_iter=opt_config.get("max_iter", 10),
            alpha=opt_config.get("alpha", 0.05),
            normalize_columns=opt_config.get("normalize_columns", False),
        )

    raise ValueError(f"Unknown optimizer type: {opt_type}")


def print_equation(target_name, feature_names, coefficients, precision=4):
    """Print an equation using fitted coefficients."""

    active_terms = []
    for coef, feature in zip(np.ravel(coefficients), feature_names):
        if abs(coef) > 1e-12:
            active_terms.append(f"{coef:.{precision}f} {feature}")

    rhs = " + ".join(active_terms).replace("+ -", "- ")
    if not rhs:
        rhs = "0"
    print(f"{target_name} = {rhs}")


def fit_sparse_system(feature_matrix, target_vector, feature_names, target_name, filename, opt_config):
    """Fit one sparse regression problem and return results."""

    optimizer = build_optimizer(opt_config)
    optimizer.fit(feature_matrix, target_vector)

    coefficients = np.asarray(optimizer.coef_)
    coefficient_tol = opt_config.get("coefficient_tol", 0.0)
    if coefficient_tol > 0:
        coefficients[np.abs(coefficients) < coefficient_tol] = 0.0
    if coefficients.ndim == 1:
        coefficients = coefficients[np.newaxis, :]

    print_equation(target_name, feature_names, coefficients[0])
    print()

    return {
        "dataset": filename.split(".")[0],
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }


def run_sindy(data, x, y, z, t, filename):
    """Run PySINDy sparse discovery for one configured dataset."""

    params = sindy_params[filename]
    
    data_arrays = sindy_library.normalize_data_arrays(data)
    lib_config = params.get("library", {})
    variable_names = lib_config.get("variable_names", sindy_library.default_variable_names(data_arrays))
    targets = params.get("targets", sindy_library.default_targets(variable_names))
    derivatives = compute_derivative_bundle(
        data_arrays if len(data_arrays) > 1 else data_arrays[0],
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=variable_names,
        max_orders=sindy_library.configured_max_deriv_order(data_arrays[0].shape, params),
    )
    crop_slices = sindy_library.build_crop_slices(data_arrays[0].shape, params.get("crop", 0))

    results = []
    feature_names_by_target = []
    for target in targets:
        target_name, features, feature_names, target_values = sindy_library.build_target_problem(
            target,
            params,
            derivatives,
            crop_slices,
            data_arrays[0].shape,
            x,
            t,
        )
        result = fit_sparse_system(
            features,
            target_values,
            feature_names,
            target_name,
            filename,
            params["optimizer"],
        )
        results.append(result)
        feature_names_by_target.append(feature_names)

    return {
        "dataset": filename.split(".")[0],
        "targets": [result["target"] for result in results],
        "coefficients": [result["coefficients"][0] for result in results],
        "features": feature_names_by_target,
    }


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            result = run_sindy(data, x, y, z, t, dataset)
            all_results.append(result)
        except Exception as error:
            print(f"Error processing {dataset}: {error}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
