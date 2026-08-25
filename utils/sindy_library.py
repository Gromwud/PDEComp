from itertools import product

import numpy as np

from data.config import COMMON_PARAMS
from utils.derivatives import (
    compute_derivative_bundle,
    derivative_name,
    get_derivative,
    numpy_gradient_derivative,
)


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


def generate_polynomial_tokens(
    fields,
    variable_names,
    degree=3,
    max_factors=2,
    include_bias=True,
):
    specs = []
    for exponents in product(range(degree + 1), repeat=len(variable_names)):
        factor_count = sum(exponent > 0 for exponent in exponents)
        if factor_count == 0:
            if include_bias:
                reference = fields[variable_names[0]]
                specs.append(("1", np.ones_like(reference, dtype=float)))
            continue
        if max_factors is not None and factor_count > max_factors:
            continue

        value = np.ones_like(fields[variable_names[0]], dtype=float)
        for variable_name, exponent in zip(variable_names, exponents):
            if exponent:
                value = value * (fields[variable_name] ** exponent)
        specs.append((monomial_name(variable_names, exponents), value))

    specs.sort(key=lambda spec: (spec[0].count(" "), spec[0]))
    return specs


def token_factor_count(token_name):
    return len(token_name.split())


def axis_order_limit(max_deriv_order, axis_name, axis_names):
    axis_index = axis_names.index(axis_name)
    return max_deriv_order[axis_index]


def generate_derivative_tokens(bundle, variable_names, axis_names, max_deriv_order):
    specs = []
    for variable_name in variable_names:
        for axis_name in axis_names:
            if axis_name not in bundle["axis_names"]:
                continue
            axis_index = bundle["axis_names"].index(axis_name)
            available_order = bundle["max_orders"][axis_index]
            axis_max_order = axis_order_limit(max_deriv_order, axis_name, bundle["axis_names"])
            for order in range(1, min(axis_max_order, available_order) + 1):
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


def default_derivative_axes(axis_names):
    return [axis_name for axis_name in axis_names if axis_name != "t"]


def feature_field(bundle, field_name):
    if field_name in bundle["variables"]:
        return bundle["variables"][field_name]["values"]

    for axis_name, axis_index, grid in bundle["axes"]:
        if field_name == axis_name:
            shape = next(iter(bundle["variables"].values()))["values"].shape
            return broadcast_axis(grid, shape, axis_index)

    raise KeyError(f"Unknown library field {field_name!r}")


def generate_feature_library(
    bundle,
    polynomial_variables,
    derivative_variables,
    derivative_axes,
    polynomial_degree=3,
    max_deriv_order=None,
    custom_tokens=None,
    coordinate_variables=None,
    max_factors_in_term=2,
    include_bias=True,
    include_polynomials=True,
    include_derivatives=True,
    include_products=True,
):
    fields = {
        variable_name: feature_field(bundle, variable_name)
        for variable_name in polynomial_variables
    }
    polynomial_specs = generate_polynomial_tokens(
        fields,
        polynomial_variables,
        degree=polynomial_degree,
        max_factors=max_factors_in_term,
        include_bias=include_bias,
    )
    derivative_specs = generate_derivative_tokens(
        bundle,
        derivative_variables,
        derivative_axes,
        max_deriv_order=max_deriv_order,
    )
    coordinate_specs = [
        (coordinate_name, feature_field(bundle, coordinate_name))
        for coordinate_name in (coordinate_variables or [])
    ]

    feature_specs = []
    if include_polynomials:
        feature_specs.extend(polynomial_specs)
    if include_derivatives:
        feature_specs.extend(derivative_specs)
    if include_products and max_factors_in_term >= 2:
        for polynomial_name, polynomial_values in polynomial_specs:
            if polynomial_name == "1":
                continue
            if token_factor_count(polynomial_name) + 1 > max_factors_in_term:
                continue
            for derivative_name_, derivative_values in derivative_specs:
                feature_specs.append((
                    f"{polynomial_name} {derivative_name_}",
                    polynomial_values * derivative_values,
                ))
        for coordinate_name, coordinate_values in coordinate_specs:
            for derivative_name_, derivative_values in derivative_specs:
                feature_specs.append((
                    f"{coordinate_name} {derivative_name_}",
                    coordinate_values * derivative_values,
                ))
    if custom_tokens:
        feature_specs.extend(custom_tokens)

    return deduplicate_feature_specs(feature_specs)


def repeated_axis_derivative_names(variable_names, axes, order):
    return {
        f"{variable_name}_{axis_name * order}"
        for variable_name in variable_names
        for axis_name in axes
    }


def second_derivative_sum_feature(bundle, variable_name, axes):
    name = " + ".join(f"{variable_name}_{axis_name}{axis_name}" for axis_name in axes)
    values = sum(
        get_derivative(bundle, variable_name, axis_name, 2)
        for axis_name in axes
    )
    return f"({name})", values


def build_ns_features(bundle, crop_slices, params, target_variable):
    lib_config = params.get("library", {})
    velocity_variables = lib_config.get("ns_velocity_variables", ["u", "v"])
    pressure_variable = lib_config.get("ns_pressure_variable", "p")
    axes = lib_config.get("ns_derivative_axes", ["x", "y"])
    polynomial_degree = lib_config.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"])
    max_deriv_order = lib_config.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"])

    velocity_fields = {
        variable_name: bundle["variables"][variable_name]["values"]
        for variable_name in velocity_variables
    }

    feature_specs = generate_polynomial_tokens(
        velocity_fields,
        velocity_variables,
        degree=polynomial_degree,
        max_factors=1,
        include_bias=True,
    )

    derivative_specs = generate_derivative_tokens(
        bundle,
        velocity_variables + [pressure_variable],
        axes,
        max_deriv_order=max_deriv_order,
    )

    first_velocity_derivatives = repeated_axis_derivative_names(
        velocity_variables,
        axes,
        order=1,
    )
    derivative_specs = [
        (name, values)
        for name, values in derivative_specs
        if name not in first_velocity_derivatives
    ]

    second_velocity_derivatives = repeated_axis_derivative_names(
        velocity_variables,
        axes,
        order=2,
    )
    derivative_specs = [
        (name, values)
        for name, values in derivative_specs
        if name not in second_velocity_derivatives
    ]

    feature_specs.extend(derivative_specs)

    for velocity_variable, axis_name in zip(velocity_variables, axes):
        velocity_values = bundle["variables"][velocity_variable]["values"]
        target_derivative = get_derivative(bundle, target_variable, axis_name, 1)
        feature_specs.append((
            f"{velocity_variable} {target_variable}_{axis_name}",
            velocity_values * target_derivative,
        ))

    for variable_name in velocity_variables:
        feature_specs.append(second_derivative_sum_feature(
            bundle,
            variable_name,
            axes,
        ))

    return build_feature_matrix(deduplicate_feature_specs(feature_specs), crop_slices)


def max_factors_in_term(lib_config):
    return int(lib_config.get(
        "equation_factors_max_number",
        COMMON_PARAMS["equation_factors_max_number"],
    ))


def library_settings(params):
    lib_config = params.get("library", {})
    return {
        "polynomial_degree": lib_config.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"]),
        "max_deriv_order": lib_config.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]),
        "max_factors_in_term": max_factors_in_term(lib_config),
        "include_bias": lib_config.get(
            "include_bias",
            COMMON_PARAMS["include_bias"],
        ),
    }


def build_configured_features(
    params,
    bundle,
    crop_slices,
    polynomial_variables=None,
    derivative_variables=None,
    derivative_axes=None,
    custom_tokens=None,
    include_polynomials=True,
    include_derivatives=True,
    include_products=True,
):
    lib_config = params.get("library", {})
    settings = library_settings(params)
    feature_specs = generate_feature_library(
        bundle=bundle,
        polynomial_variables=(
            lib_config.get("polynomial_variables", ["u"])
            if polynomial_variables is None
            else polynomial_variables
        ),
        derivative_variables=(
            lib_config.get("derivative_variables", ["u"])
            if derivative_variables is None
            else derivative_variables
        ),
        derivative_axes=(
            lib_config.get("derivative_axes", default_derivative_axes(bundle["axis_names"]))
            if derivative_axes is None
            else derivative_axes
        ),
        polynomial_degree=settings["polynomial_degree"],
        max_deriv_order=settings.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]),
        custom_tokens=custom_tokens,
        coordinate_variables=lib_config.get("coordinate_variables", []),
        max_factors_in_term=settings["max_factors_in_term"],
        include_bias=settings["include_bias"],
        include_polynomials=include_polynomials,
        include_derivatives=include_derivatives,
        include_products=include_products,
    )
    return build_feature_matrix(feature_specs, crop_slices)


def derivative_token_value(bundle, token_name):
    if "_" not in token_name:
        return bundle["variables"][token_name]["values"]

    variable_name, suffix = token_name.split("_", 1)
    if not suffix or len(set(suffix)) != 1:
        raise ValueError(f"Cannot parse derivative token {token_name!r}")
    return get_derivative(bundle, variable_name, suffix[0], len(suffix))


def grid_values(values, shape, axis):
    if len(shape) == 1:
        return np.asarray(values, dtype=float)
    return broadcast_axis(values, shape, axis=axis)


def safe_divide_by_coordinate(values, coordinate_values):
    return np.divide(
        values,
        coordinate_values,
        out=np.zeros_like(values, dtype=float),
        where=np.abs(coordinate_values) > 1e-12,
    )


def build_custom_tokens(token_names, bundle, data_shape, x, t):
    if not token_names:
        return []

    specs = []
    u = bundle["variables"].get("u", {}).get("values")
    for token_name in token_names:
        if token_name == "t":
            specs.append(("t", grid_values(t, data_shape, axis=0)))
        elif token_name == "x":
            specs.append(("x", grid_values(x, data_shape, axis=1)))
        elif token_name == "t^2":
            t_values = grid_values(t, data_shape, axis=0)
            specs.append(("t^2", t_values ** 2))
        elif token_name == "u_t sin(2 t)":
            u_t = get_derivative(bundle, "u", "t", 1)
            t_values = grid_values(t, data_shape, axis=0)
            specs.append(("u_t sin(2 t)", u_t * np.sin(2 * t_values)))
        elif token_name == "sin(t)":
            specs.append(("sin(t)", np.sin(grid_values(t, data_shape, axis=0))))
        elif token_name == "cos(t)":
            specs.append(("cos(t)", np.cos(grid_values(t, data_shape, axis=0))))
        elif token_name == "sin(2 t)":
            specs.append(("sin(2 t)", np.sin(2 * grid_values(t, data_shape, axis=0))))
        elif token_name == "cos(2 t)":
            specs.append(("cos(2 t)", np.cos(2 * grid_values(t, data_shape, axis=0))))
        elif token_name == "(1/x) u":
            x_grid = grid_values(x, data_shape, axis=1)
            specs.append(("(1/x) u", safe_divide_by_coordinate(u, x_grid)))
        elif token_name == "(1/x) u_x":
            x_grid = grid_values(x, data_shape, axis=1)
            specs.append((
                "(1/x) u_x",
                safe_divide_by_coordinate(get_derivative(bundle, "u", "x", 1), x_grid),
            ))
        elif token_name == "d_x(u u_x)":
            u_x = get_derivative(bundle, "u", "x", 1)
            specs.append(("d_x(u u_x)", numpy_gradient_derivative(u * u_x, x[1] - x[0], axis=1, order=1)))
        elif token_name == "sin(x)":
            specs.append(("sin(x)", np.sin(grid_values(x, data_shape, axis=1))))
        elif token_name == "sin(x) cos(t)":
            specs.append((
                "sin(x) cos(t)",
                np.sin(grid_values(x, data_shape, axis=1)) * np.cos(grid_values(t, data_shape, axis=0)),
            ))
        elif token_name == "cos(x) sin(t)":
            specs.append((
                "cos(x) sin(t)",
                np.cos(grid_values(x, data_shape, axis=1)) * np.sin(grid_values(t, data_shape, axis=0)),
            ))
        else:
            specs.append((token_name, derivative_token_value(bundle, token_name)))
    return specs


def default_variable_names(data_arrays):
    if len(data_arrays) == 1:
        return ["u"]
    return [f"x{idx}" for idx in range(len(data_arrays))]


def normalize_data_arrays(data):
    if isinstance(data, list):
        return [np.asarray(item, dtype=float) for item in data]
    return [np.asarray(data, dtype=float)]


def configured_max_deriv_order(data_shape, params):
    lib_config = params.get("library", {})
    return tuple(lib_config.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"])[:len(data_shape)])


def default_targets(variable_names):
    return [
        {
            "name": f"{variable_name}_t",
            "variable": variable_name,
            "axis": "t",
            "order": 1,
        }
        for variable_name in variable_names
    ]


def build_target_features(target, params, bundle, crop_slices, data_shape, x, t):
    lib_config = params.get("library", {})
    if "feature_tokens" in target:
        feature_specs = [
            (token_name, derivative_token_value(bundle, token_name))
            for token_name in target["feature_tokens"]
        ]
        return build_feature_matrix(feature_specs, crop_slices)

    if lib_config.get("type") == "navier_stokes":
        return build_ns_features(
            bundle,
            crop_slices,
            params,
            target_variable=target["variable"],
        )

    custom_tokens = build_custom_tokens(
        lib_config.get("custom_tokens", []),
        bundle,
        data_shape,
        x,
        t,
    )
    return build_configured_features(
        params=params,
        bundle=bundle,
        crop_slices=crop_slices,
        custom_tokens=custom_tokens,
    )


def target_name_from_config(target, axis_names):
    if "name" in target:
        return target["name"]
    target_axis = target.get("axis", "t")
    target_order = target.get("order", 1)
    orders = tuple(target_order if axis_name == target_axis else 0 for axis_name in axis_names)
    return derivative_name(target["variable"], orders, axis_names)


def remove_target_from_features(features, feature_names, target_name):
    def uses_target(feature_name):
        tokens = (
            feature_name
            .replace("(", " ")
            .replace(")", " ")
            .replace("+", " ")
            .split()
        )
        return target_name in tokens

    keep_indexes = [
        index for index, feature_name in enumerate(feature_names)
        if not uses_target(feature_name)
    ]
    if len(keep_indexes) == len(feature_names):
        return features, feature_names
    return features[:, keep_indexes], [feature_names[index] for index in keep_indexes]


def build_target_problem(target, params, bundle, crop_slices, data_shape, x, t):
    target_name = target_name_from_config(target, bundle["axis_names"])
    target_axis = target.get("axis", "t")
    target_order = target.get("order", 1)
    features, feature_names = build_target_features(
        target,
        params,
        bundle,
        crop_slices,
        data_shape,
        x,
        t,
    )
    features, feature_names = remove_target_from_features(
        features,
        feature_names,
        target_name,
    )
    target_values = crop_and_flatten(
        get_derivative(bundle, target["variable"], target_axis, target_order),
        crop_slices,
    )
    return target_name, features, feature_names, target_values
