from itertools import product

import numpy as np


def numpy_gradient_derivative(values, step, axis, order=1):
    derivative = np.asarray(values, dtype=float)
    for _ in range(order):
        derivative = np.gradient(derivative, step, axis=axis)
    return derivative


def get_data_axes(data, x, y, z, t):
    values = np.asarray(data)
    if values.ndim == 1:
        return [("t", 0, np.asarray(t, dtype=float))]
    if values.ndim == 2:
        return [("t", 0, np.asarray(t, dtype=float)), ("x", 1, np.asarray(x, dtype=float))]
    if values.ndim == 3:
        return [
            ("t", 0, np.asarray(t, dtype=float)),
            ("y", 1, np.asarray(y, dtype=float)),
            ("x", 2, np.asarray(x, dtype=float)),
        ]
    raise ValueError(f"Unsupported data dimensionality for derivatives: {values.ndim}")


def get_axis_steps(axes):
    steps = []
    for axis_name, _, grid in axes:
        if grid is None or len(grid) < 2:
            raise ValueError(f"Cannot compute derivatives: axis {axis_name!r} has no grid step")
        steps.append(float(grid[1] - grid[0]))
    return steps


def normalize_max_orders(max_orders, ndim, default_spatial_order=4, default_time_order=2):
    if max_orders is None:
        return tuple([default_time_order] + [default_spatial_order] * (ndim - 1))
    if isinstance(max_orders, int):
        return tuple([max_orders] * ndim)
    if len(max_orders) > ndim:
        return tuple(max_orders[:ndim])
    if len(max_orders) != ndim:
        raise ValueError(f"Expected {ndim} derivative orders, got {max_orders}")
    return tuple(max_orders)


def derivative_name(variable_name, orders, axis_names):
    suffix = "".join(axis_name * order for axis_name, order in zip(axis_names, orders))
    return f"{variable_name}_{suffix}" if suffix else variable_name


def compute_multi_derivative(values, axes, orders):
    derivative = np.asarray(values, dtype=float)
    steps = get_axis_steps(axes)
    for (_, axis, _), step, order in zip(axes, steps, orders):
        derivative = numpy_gradient_derivative(derivative, step, axis=axis, order=order)
    return derivative


def derivative_multiindices(max_orders, include_identity=False):
    ranges = [range(order + 1) for order in max_orders]
    for orders in product(*ranges):
        if not include_identity and all(order == 0 for order in orders):
            continue
        yield orders


def compute_derivative_bundle(data, x, y, z, t, variable_names=None, max_orders=None):
    if isinstance(data, list):
        arrays = [np.asarray(item, dtype=float) for item in data]
    else:
        arrays = [np.asarray(data, dtype=float)]

    if variable_names is None:
        variable_names = [f"u{idx}" if len(arrays) > 1 else "u" for idx in range(len(arrays))]

    axes = get_data_axes(arrays[0], x, y, z, t)
    axis_names = [axis_name for axis_name, _, _ in axes]
    max_orders = normalize_max_orders(max_orders, len(axes))

    variables = {}
    for variable_name, values in zip(variable_names, arrays):
        derivatives = {}
        for orders in derivative_multiindices(max_orders, include_identity=True):
            derivatives[orders] = compute_multi_derivative(values, axes, orders)
        variables[variable_name] = {
            "values": values,
            "derivatives": derivatives,
        }

    return {
        "axes": axes,
        "axis_names": axis_names,
        "max_orders": max_orders,
        "variables": variables,
    }


def axis_orders(bundle, axis_name, order):
    if axis_name not in bundle["axis_names"]:
        raise ValueError(f"Axis {axis_name!r} is not available in this derivative bundle")
    orders = [0] * len(bundle["axis_names"])
    orders[bundle["axis_names"].index(axis_name)] = order
    return tuple(orders)


def get_derivative(bundle, variable_name, axis_name, order):
    return bundle["variables"][variable_name]["derivatives"][axis_orders(bundle, axis_name, order)]


def get_multi_derivative(bundle, variable_name, orders):
    return bundle["variables"][variable_name]["derivatives"][tuple(orders)]


def build_epde_derivatives(data, x, y, z, t, variable_names, max_deriv_order):
    if isinstance(data, list):
        arrays = [np.asarray(item, dtype=float) for item in data]
    else:
        arrays = [np.asarray(data, dtype=float)]

    axes = get_data_axes(arrays[0], x, y, z, t)
    max_orders = normalize_max_orders(max_deriv_order, len(axes))
    epde_derivs = []

    for values in arrays:
        derivatives = [
            compute_multi_derivative(values, axes, orders)
            for orders in derivative_multiindices(max_orders, include_identity=False)
        ]
        epde_derivs.append(np.stack(derivatives, axis=-1))

    return epde_derivs
