import json
from datetime import datetime
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pysindy as ps

sys.path.append(str(Path().absolute()))

from data.config import sindy_params
from data.dataloader import load_data
from data.derivatives import (
    compute_derivative_bundle,
    derivative_name,
    get_derivative,
    numpy_gradient_derivative,
)


DATA_DIR = Path("data")
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
GENERATED_MANUAL_DATASETS = {
    "burgers_data.mat",
    "ac_data.npy",
    "kdv_data.mat",
    "ks_data.mat",
    "burgers_sln_100_data.csv",
}


def save_combined_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as handle:
        json.dump([results], handle, indent=2)


def build_pde_library(lib_config, x):
    return ps.PDELibrary(
        derivative_order=lib_config.get("derivative_order", 3),
        spatial_grid=x,
        include_bias=lib_config.get("pde_include_bias", True),
        function_library=ps.PolynomialLibrary(
            degree=lib_config.get("poly_degree", 2),
            include_bias=lib_config.get("poly_include_bias", False),
        ),
        differentiation_method=ps.FiniteDifference,
    )


def build_library(lib_config, x):
    lib_type = lib_config.get("type", "polynomial")

    if lib_type == "polynomial":
        return ps.PolynomialLibrary(
            degree=lib_config.get("degree", 2),
            include_bias=lib_config.get("include_bias", True),
        )

    if lib_type == "pde":
        return build_pde_library(lib_config, x)

    if lib_type == "pde_custom_concat":
        base_lib = build_pde_library(lib_config, x)
        functions = [lambda val: 1 / val]
        function_names = [lambda val: "1/" + val]
        custom_lib = ps.CustomLibrary(
            library_functions=functions,
            function_names=function_names,
        ) * base_lib
        return ps.ConcatLibrary([custom_lib, base_lib])

    if lib_type == "poly_and_fourier":
        poly_lib = ps.PolynomialLibrary(
            degree=lib_config.get("poly_degree", 2),
            include_bias=lib_config.get("poly_include_bias", True),
        )
        trig_lib = ps.FourierLibrary(n_frequencies=lib_config.get("n_frequencies", 1))
        return poly_lib + trig_lib

    raise ValueError(f"Unknown library type: {lib_type}")


def build_optimizer(opt_config):
    opt_type = opt_config.get("type", "STLSQ")

    if opt_type == "STLSQ":
        return ps.STLSQ(
            threshold=opt_config.get("threshold", 0.1),
            alpha=opt_config.get("alpha", 0.05),
            normalize_columns=opt_config.get("normalize_columns", False),
        )

    if opt_type == "SR3":
        return ps.SR3(
            max_iter=opt_config.get("max_iter", 30),
            tol=opt_config.get("tol", 1e-5),
            normalize_columns=opt_config.get("normalize_columns", False),
            reg_weight_lam=opt_config.get("reg_weight_lam", opt_config.get("threshold", 0.1)),
            regularizer=opt_config.get("regularizer", opt_config.get("thresholder", "L0")),
        )

    raise ValueError(f"Unknown optimizer type: {opt_type}")


def prepare_standard_data(data, x, t, filename):
    if isinstance(data, list):
        data = np.array(data)
    if len(data.shape) == 1:
        data = data.T.reshape(len(t), 1)
    elif filename in ["lotka_data.npy", "lorenz_data.npy", "ODE_simple_discovery"]:
        data = data.T
    elif len(data.shape) == 2:
        data = data.T.reshape(len(x), len(t), 1)
    return data


def finite_difference(data, step, axis, order=1):
    return numpy_gradient_derivative(data, step, axis=axis, order=order)


def build_crop_slices(shape, crop):
    if crop <= 0:
        return tuple(slice(None) for _ in shape)
    return tuple(slice(crop, dim - crop) for dim in shape)


def crop_and_flatten(values, crop_slices):
    values = np.asarray(values, dtype=float)
    return values[crop_slices].ravel()


def build_feature_matrix(feature_specs, crop_slices):
    feature_names = [name for name, _ in feature_specs]
    features = np.column_stack([
        crop_and_flatten(values, crop_slices) for _, values in feature_specs
    ])
    return features, feature_names


def broadcast_axis(values, shape, axis):
    reshape = [1] * len(shape)
    reshape[axis] = len(values)
    return np.broadcast_to(np.asarray(values, dtype=float).reshape(reshape), shape)


def monomial_name(variable_names, exponents):
    parts = []
    for variable_name, exponent in zip(variable_names, exponents):
        if exponent == 0:
            continue
        if exponent == 1:
            parts.append(variable_name)
        else:
            parts.append(f"{variable_name}^{exponent}")
    return " ".join(parts) if parts else "1"


def generate_polynomial_tokens(fields, variable_names, degree=3, include_bias=True):
    specs = []
    for exponents in product(range(degree + 1), repeat=len(variable_names)):
        total_degree = sum(exponents)
        if total_degree == 0:
            if include_bias:
                reference = fields[variable_names[0]]
                specs.append(("1", np.ones_like(reference, dtype=float)))
            continue
        if total_degree > degree:
            continue

        value = np.ones_like(fields[variable_names[0]], dtype=float)
        for variable_name, exponent in zip(variable_names, exponents):
            if exponent:
                value = value * (fields[variable_name] ** exponent)
        specs.append((monomial_name(variable_names, exponents), value))

    specs.sort(key=lambda spec: (spec[0].count(" "), spec[0]))
    return specs


def generate_derivative_tokens(bundle, variable_names, axis_names, max_order=4):
    specs = []
    for variable_name in variable_names:
        for axis_name in axis_names:
            if axis_name not in bundle["axis_names"]:
                continue
            axis_index = bundle["axis_names"].index(axis_name)
            available_order = bundle["max_orders"][axis_index]
            for order in range(1, min(max_order, available_order) + 1):
                orders = [0] * len(bundle["axis_names"])
                orders[axis_index] = order
                name = derivative_name(variable_name, tuple(orders), bundle["axis_names"])
                specs.append((name, get_derivative(bundle, variable_name, axis_name, order)))
    return specs


def deduplicate_feature_specs(feature_specs):
    deduplicated = []
    seen = set()
    for name, values in feature_specs:
        if name in seen:
            continue
        deduplicated.append((name, values))
        seen.add(name)
    return deduplicated


def generate_manual_library(
    bundle,
    polynomial_variables,
    derivative_variables,
    derivative_axes,
    polynomial_degree=3,
    derivative_order=4,
    custom_tokens=None,
    include_bias=True,
    include_polynomials=True,
    include_derivatives=True,
    include_products=True,
):
    fields = {
        variable_name: bundle["variables"][variable_name]["values"]
        for variable_name in polynomial_variables
    }
    polynomial_specs = generate_polynomial_tokens(
        fields,
        polynomial_variables,
        degree=polynomial_degree,
        include_bias=include_bias,
    )
    derivative_specs = generate_derivative_tokens(
        bundle,
        derivative_variables,
        derivative_axes,
        max_order=derivative_order,
    )

    feature_specs = []
    if include_polynomials:
        feature_specs.extend(polynomial_specs)
    if include_derivatives:
        feature_specs.extend(derivative_specs)
    if include_products:
        for polynomial_name, polynomial_values in polynomial_specs:
            if polynomial_name == "1":
                continue
            for derivative_name_, derivative_values in derivative_specs:
                feature_specs.append((
                    f"{polynomial_name} {derivative_name_}",
                    polynomial_values * derivative_values,
                ))
    if custom_tokens:
        feature_specs.extend(custom_tokens)

    return deduplicate_feature_specs(feature_specs)


def build_generated_manual_features(
    bundle,
    crop_slices,
    polynomial_variables,
    derivative_variables,
    derivative_axes,
    polynomial_degree=3,
    derivative_order=4,
    custom_tokens=None,
    include_bias=True,
    include_polynomials=True,
    include_derivatives=True,
    include_products=True,
):
    feature_specs = generate_manual_library(
        bundle=bundle,
        polynomial_variables=polynomial_variables,
        derivative_variables=derivative_variables,
        derivative_axes=derivative_axes,
        polynomial_degree=polynomial_degree,
        derivative_order=derivative_order,
        custom_tokens=custom_tokens,
        include_bias=include_bias,
        include_polynomials=include_polynomials,
        include_derivatives=include_derivatives,
        include_products=include_products,
    )
    return build_feature_matrix(feature_specs, crop_slices)


def print_sparse_equation(target_name, feature_names, coefficients, precision=4):
    active_terms = []
    for coef, feature in zip(np.ravel(coefficients), feature_names):
        if abs(coef) > 1e-12:
            active_terms.append(f"{coef:.{precision}f} {feature}")

    rhs = " + ".join(active_terms).replace("+ -", "- ")
    if not rhs:
        rhs = "0"
    print(f"{target_name} = {rhs}")


def fit_manual_system(feature_matrix, target_vector, feature_names, target_name, filename, opt_config):
    optimizer = build_optimizer(opt_config)
    optimizer.fit(feature_matrix, target_vector)

    coefficients = np.asarray(optimizer.coef_)
    coefficient_tol = opt_config.get("coefficient_tol", 0.0)
    if coefficient_tol > 0:
        coefficients[np.abs(coefficients) < coefficient_tol] = 0.0
    if coefficients.ndim == 1:
        coefficients = coefficients[np.newaxis, :]

    print_sparse_equation(target_name, feature_names, coefficients[0])

    return {
        "dataset": filename.split(".")[0],
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }


def run_manual_dataset(data, x, y, z, t, filename, params):
    crop = params.get("crop", 0)

    if isinstance(data, list):
        data = np.array(data)

    if filename == "ode_data.npy":
        u = np.asarray(data, dtype=float).reshape(-1)
        derivatives = compute_derivative_bundle(u, x=None, y=None, z=None, t=t, variable_names=["u"], max_orders=(2,))
        u_t = get_derivative(derivatives, "u", "t", 1)
        u_tt = get_derivative(derivatives, "u", "t", 2)
        crop_slices = build_crop_slices(u.shape, crop)[0]
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["t"],
            polynomial_degree=3,
            derivative_order=1,
            custom_tokens=[
                ("t", t),
                ("t^2", t ** 2),
                ("u_t sin(2 t)", u_t * np.sin(2 * t)),
            ],
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_tt, crop_slices),
            feature_names,
            "u_tt",
            filename,
            params["optimizer"],
        )

    if filename == "vdp_data.npy":
        u = np.asarray(data, dtype=float).reshape(-1)
        derivatives = compute_derivative_bundle(u, x=None, y=None, z=None, t=t, variable_names=["u"], max_orders=(2,))
        u_tt = get_derivative(derivatives, "u", "t", 2)
        crop_slices = build_crop_slices(u.shape, crop)[0]
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["t"],
            polynomial_degree=3,
            derivative_order=1,
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_tt, crop_slices),
            feature_names,
            "u_tt",
            filename,
            params["optimizer"],
        )

    if filename == "ODE_simple_discovery":
        u = np.asarray(data[0], dtype=float).reshape(-1)
        derivatives = compute_derivative_bundle(u, x=None, y=None, z=None, t=t, variable_names=["u"], max_orders=(1,))
        u_t = get_derivative(derivatives, "u", "t", 1)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=slice(None),
            polynomial_variables=["u"],
            derivative_variables=[],
            derivative_axes=["t"],
            polynomial_degree=3,
            derivative_order=0,
            custom_tokens=[
                ("sin(t)", np.sin(t)),
                ("cos(t)", np.cos(t)),
                ("sin(2 t)", np.sin(2 * t)),
                ("cos(2 t)", np.cos(2 * t)),
            ],
        )
        return fit_manual_system(features, u_t, feature_names, "u_t", filename, params["optimizer"])

    if filename == "ns_data.mat":
        u = np.asarray(data[0], dtype=float)
        v = np.asarray(data[1], dtype=float)
        p = np.asarray(data[2], dtype=float)
        derivatives = compute_derivative_bundle(
            [u, v, p],
            x=x,
            y=y,
            z=z,
            t=t,
            variable_names=["u", "v", "p"],
            max_orders=(1, 4, 4),
        )
        crop_slices = build_crop_slices(u.shape, crop)
        u_t = get_derivative(derivatives, "u", "t", 1)
        v_t = get_derivative(derivatives, "v", "t", 1)
        u_x = get_derivative(derivatives, "u", "x", 1)
        u_xx = get_derivative(derivatives, "u", "x", 2)
        u_yy = get_derivative(derivatives, "u", "y", 2)
        v_xx = get_derivative(derivatives, "v", "x", 2)
        v_yy = get_derivative(derivatives, "v", "y", 2)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u", "v", "p"],
            derivative_variables=["u", "v", "p"],
            derivative_axes=["x", "y"],
            polynomial_degree=1,
            derivative_order=2,
            custom_tokens=[
                ("(u_xx + u_yy)", u_xx + u_yy),
                ("(v_xx + v_yy)", v_xx + v_yy),
            ],
        )

        optimizer = build_optimizer(params["optimizer"])
        optimizer.fit(features, crop_and_flatten(u_t, crop_slices))
        u_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("u_t", feature_names, u_coefficients)

        optimizer = build_optimizer(params["optimizer"])
        optimizer.fit(features, crop_and_flatten(v_t, crop_slices))
        v_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("v_t", feature_names, v_coefficients)

        optimizer = build_optimizer(params["optimizer"])
        optimizer.fit(features, crop_and_flatten(u_x, crop_slices))
        continuity_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("u_x", feature_names, continuity_coefficients)

        return {
            "dataset": filename.split(".")[0],
            "targets": ["u_t", "v_t", "u_x"],
            "coefficients": [
                u_coefficients.tolist(),
                v_coefficients.tolist(),
                continuity_coefficients.tolist(),
            ],
            "features": [
                feature_names,
                feature_names,
                feature_names,
            ],
        }

    u = np.asarray(data, dtype=float)
    derivatives = compute_derivative_bundle(u, x=x, y=y, z=z, t=t, variable_names=["u"], max_orders=(2, 4))
    crop_slices = build_crop_slices(u.shape, crop)

    if filename in GENERATED_MANUAL_DATASETS:
        u_t = get_derivative(derivatives, "u", "t", 1)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["x"],
            polynomial_degree=3,
            derivative_order=4,
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_t, crop_slices),
            feature_names,
            "u_t",
            filename,
            params["optimizer"],
        )

    if filename == "wave_data.csv":
        u_tt = get_derivative(derivatives, "u", "t", 2)
        u_t = get_derivative(derivatives, "u", "t", 1)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["x"],
            polynomial_degree=3,
            derivative_order=4,
            custom_tokens=[("u_t", u_t)],
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_tt, crop_slices),
            feature_names,
            "u_tt",
            filename,
            params["optimizer"],
        )

    if filename == "pde_divide_data.npy":
        u_t = get_derivative(derivatives, "u", "t", 1)
        u_x = get_derivative(derivatives, "u", "x", 1)
        u_xx = get_derivative(derivatives, "u", "x", 2)
        x_grid = broadcast_axis(x, u.shape, axis=1)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["x"],
            polynomial_degree=3,
            derivative_order=4,
            custom_tokens=[
                ("(1/x) u", u / x_grid),
                ("(1/x) u_x", u_x / x_grid),
                ("x u_x", x_grid * u_x),
                ("x u_xx", x_grid * u_xx),
            ],
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_t, crop_slices),
            feature_names,
            "u_t",
            filename,
            params["optimizer"],
        )

    if filename == "pde_compound_data.npy":
        u_t = get_derivative(derivatives, "u", "t", 1)
        u_x = get_derivative(derivatives, "u", "x", 1)
        nonlinear_derivative = numpy_gradient_derivative(u * u_x, x[1] - x[0], axis=1, order=1)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["x"],
            polynomial_degree=3,
            derivative_order=4,
            custom_tokens=[("d_x(u u_x)", nonlinear_derivative)],
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_t, crop_slices),
            feature_names,
            "u_t",
            filename,
            params["optimizer"],
        )

    if filename == "kdv_periodic_data.npy":
        u_t = get_derivative(derivatives, "u", "t", 1)
        x_grid = broadcast_axis(x, u.shape, axis=1)
        t_grid = broadcast_axis(t, u.shape, axis=0)
        features, feature_names = build_generated_manual_features(
            bundle=derivatives,
            crop_slices=crop_slices,
            polynomial_variables=["u"],
            derivative_variables=["u"],
            derivative_axes=["x"],
            polynomial_degree=3,
            derivative_order=4,
            custom_tokens=[
                ("sin(x)", np.sin(x_grid)),
                ("cos(t)", np.cos(t_grid)),
                ("sin(x) cos(t)", np.sin(x_grid) * np.cos(t_grid)),
                ("cos(x) sin(t)", np.cos(x_grid) * np.sin(t_grid)),
            ],
        )
        return fit_manual_system(
            features,
            crop_and_flatten(u_t, crop_slices),
            feature_names,
            "u_t",
            filename,
            params["optimizer"],
        )

    raise ValueError(f"Unknown manual mode for dataset: {filename}")


def run_sindy(data, x, y, z, t, filename):
    params = sindy_params[filename]

    if params.get("manual_mode") or filename in GENERATED_MANUAL_DATASETS:
        return run_manual_dataset(data, x, y, z, t, filename, params)

    data = prepare_standard_data(data, x, t, filename)
    x_dot = None
    if filename in ["lotka_data.npy", "lorenz_data.npy"]:
        x_dot = numpy_gradient_derivative(data, t[1] - t[0], axis=0, order=1)

    if params.get("preprocess", {}).get("moveaxis", False):
        data = np.moveaxis(data, 0, -1)
        if x_dot is not None:
            x_dot = np.moveaxis(x_dot, 0, -1)

    library = build_library(params["library"], x)
    optimizer = build_optimizer(params["optimizer"])

    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    if x_dot is None:
        model.fit(data, t=t[1] - t[0])
    else:
        model.fit(data, t=t[1] - t[0], x_dot=x_dot)
    model.print(precision=4)

    return {
        "dataset": filename.split(".")[0],
        "coefficients": model.coefficients().tolist(),
        "features": model.get_feature_names(),
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
