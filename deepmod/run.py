import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path().absolute() / "deepmod" / "deepymod" / "src"))
sys.path.append(str(Path().absolute()))

from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.constraint import LeastSquares
from deepymod.model.deepmod import Library
from deepymod.model.func_approx import NN, Siren
from deepymod.model.sparse_estimators import PDEFIND, Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic

from data.config import COMMON_PARAMS, DEEPMOD_DATASETS, DEEPMOD_DEFAULTS, deepmod_params, sindy_params
from data.derivatives import compute_derivative_bundle, derivative_name, numpy_gradient_derivative
from data.dataloader import load_data


RESULTS_DIR = Path("results/deepmod")
DATASETS = DEEPMOD_DATASETS


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_terms(feature_names, coefficients, tol=1e-12):
    terms = []
    for name, coefficient in zip(feature_names, coefficients):
        if abs(coefficient) <= tol:
            continue
        terms.append(f"{coefficient:.4f} {name}")
    return " + ".join(terms).replace("+ -", "- ") or "0"


def axis_names_for_mode(data_mode):
    if data_mode in {"ode_scalar", "ode_system"}:
        return ["t"]
    if data_mode == "pde_1d":
        return ["t", "x"]
    if data_mode == "pde_2d_system":
        return ["t", "y", "x"]
    raise ValueError(f"Unknown DeepMoD data mode: {data_mode}")


def axis_to_coord_index_for_mode(data_mode):
    if data_mode in {"ode_scalar", "ode_system"}:
        return {"t": 0}
    if data_mode == "pde_1d":
        return {"t": 0, "x": 1}
    if data_mode == "pde_2d_system":
        return {"t": 0, "x": 1, "y": 2}
    raise ValueError(f"Unknown DeepMoD data mode: {data_mode}")


def target_name(target, axis_names):
    if "name" in target:
        return target["name"]
    orders = tuple(
        target.get("order", 1) if axis == target.get("axis", "t") else 0
        for axis in axis_names
    )
    return derivative_name(target["variable"], orders, axis_names)


def default_targets(variable_names):
    return [
        {"name": f"{variable}_t", "variable": variable, "axis": "t", "order": 1}
        for variable in variable_names
    ]


def max_factors(lib_config):
    value = lib_config.get("equation_factors_max_number", COMMON_PARAMS["equation_factors_max_number"])
    if isinstance(value, dict):
        return max(value.get("factors_num", [COMMON_PARAMS["equation_factors_max_number"]]))
    return int(value)


def axis_limit(max_deriv_order, axis, axis_names):
    if isinstance(max_deriv_order, int):
        return max_deriv_order
    return max_deriv_order[axis_names.index(axis)]


def token_count(name):
    return len(name.split())


class LibraryState:
    def __init__(self, coords, variable_names, axis_names, coord_index, fixed_values, sample_indices):
        self.coords = coords
        self.variable_names = variable_names
        self.axis_names = axis_names
        self.coord_index = coord_index
        self.fixed_values = fixed_values
        self.sample_indices = sample_indices

    def field(self, name):
        if name in self.fixed_values:
            return self.fixed(name)
        if name in self.coord_index:
            return self.coords[:, self.coord_index[name] : self.coord_index[name] + 1]
        raise KeyError(f"Unknown field {name!r}")

    def fixed(self, name):
        values = self.fixed_values[name].to(self.coords.device)
        return values[self.sample_indices]

    def has_fixed(self, name):
        return name in self.fixed_values

    def deriv(self, variable, axis, order):
        if order == 0:
            return self.field(variable)
        orders = [0] * len(self.axis_names)
        orders[self.axis_names.index(axis)] = order
        return self.fixed(derivative_name(variable, tuple(orders), self.axis_names))


def parse_derivative_token(name, state):
    if state.has_fixed(name):
        return state.fixed(name)
    if name == "1":
        return torch.ones_like(state.field(state.variable_names[0]))
    if "_" not in name:
        return state.field(name)
    variable, suffix = name.split("_", 1)
    if not suffix or len(set(suffix)) != 1:
        raise ValueError(f"Cannot parse derivative token {name!r}")
    return state.deriv(variable, suffix[0], len(suffix))


def custom_token(name, state):
    if state.has_fixed(name):
        return state.fixed(name)
    if name in {"t", "x", "y"}:
        return state.field(name)
    if name == "sin(t)":
        return torch.sin(state.field("t"))
    if name == "cos(t)":
        return torch.cos(state.field("t"))
    if name == "sin(x)":
        return torch.sin(state.field("x"))
    if name == "cos(x)":
        return torch.cos(state.field("x"))
    if name == "u_t sin(2 t)":
        return state.deriv("u", "t", 1) * torch.sin(2 * state.field("t"))
    if name == "sin(x) cos(t)":
        return torch.sin(state.field("x")) * torch.cos(state.field("t"))
    if name == "cos(x) sin(t)":
        return torch.cos(state.field("x")) * torch.sin(state.field("t"))
    if name == "(1/x) u":
        return state.field("u") / torch.where(torch.abs(state.field("x")) > 1e-12, state.field("x"), torch.ones_like(state.field("x")))
    if name == "(1/x) u_x":
        u_x = state.deriv("u", "x", 1)
        return u_x / torch.where(torch.abs(state.field("x")) > 1e-12, state.field("x"), torch.ones_like(state.field("x")))
    return parse_derivative_token(name, state)


def polynomial_specs(state, variables, degree, include_bias, factor_limit):
    specs = []
    reference = state.field(variables[0])
    if include_bias:
        specs.append(("1", torch.ones_like(reference)))
    for exponents in product(range(degree + 1), repeat=len(variables)):
        if sum(exponents) == 0:
            continue
        if sum(power > 0 for power in exponents) > factor_limit:
            continue
        value = torch.ones_like(reference)
        parts = []
        for variable, power in zip(variables, exponents):
            if power:
                value = value * state.field(variable) ** power
                parts.append(variable if power == 1 else f"{variable}^{power}")
        specs.append((" ".join(parts), value))
    return specs


def derivative_specs(state, variables, axes, axis_names, max_order):
    specs = []
    for variable in variables:
        for axis in axes:
            if axis not in axis_names:
                continue
            for order in range(1, axis_limit(max_order, axis, axis_names) + 1):
                orders = [0] * len(axis_names)
                orders[axis_names.index(axis)] = order
                specs.append((derivative_name(variable, tuple(orders), axis_names), state.deriv(variable, axis, order)))
    return specs


def deduplicate(specs):
    seen = set()
    result = []
    for name, value in specs:
        if name not in seen:
            result.append((name, value))
            seen.add(name)
    return result


def uses_target(feature_name, target):
    return target in feature_name.replace("(", " ").replace(")", " ").replace("+", " ").split()


def configured_specs(state, params, axis_names):
    lib_config = params["sindy_config"].get("library", {})
    variables = lib_config.get("polynomial_variables", lib_config.get("variable_names", ["u"]))
    deriv_variables = lib_config.get("derivative_variables", lib_config.get("variable_names", ["u"]))
    axes = lib_config.get("derivative_axes", [axis for axis in axis_names if axis != "t"])
    degree = lib_config.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"])
    deriv_order = lib_config.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"])
    factors = max_factors(lib_config)
    include_bias = lib_config.get("include_bias", COMMON_PARAMS["include_bias"])

    polys = polynomial_specs(state, variables, degree, include_bias, factors)
    derivs = derivative_specs(state, deriv_variables, axes, axis_names, deriv_order)

    specs = []
    if lib_config.get("include_polynomials", True):
        specs.extend(polys)
    if lib_config.get("include_derivatives", True):
        specs.extend(derivs)
    if lib_config.get("include_products", True) and factors >= 2:
        for poly_name, poly_value in polys:
            if poly_name == "1":
                continue
            if token_count(poly_name) + 1 > factors:
                continue
            for deriv_name, deriv_value in derivs:
                specs.append((f"{poly_name} {deriv_name}", poly_value * deriv_value))
    for name in lib_config.get("custom_tokens", []):
        specs.append((name, custom_token(name, state)))
    return deduplicate(specs)


def ns_specs(state, target_variable):
    u = state.field("u")
    v = state.field("v")
    p_x = state.deriv("p", "x", 1)
    p_y = state.deriv("p", "y", 1)
    p_xx = state.deriv("p", "x", 2)
    p_yy = state.deriv("p", "y", 2)
    u_x = state.deriv("u", "x", 1)
    u_y = state.deriv("u", "y", 1)
    v_x = state.deriv("v", "x", 1)
    v_y = state.deriv("v", "y", 1)
    u_lap = state.deriv("u", "x", 2) + state.deriv("u", "y", 2)
    v_lap = state.deriv("v", "x", 2) + state.deriv("v", "y", 2)
    if target_variable == "u":
        return deduplicate([
            ("1", torch.ones_like(u)), ("u", u), ("v", v),
            ("p_x", p_x), ("p_xx", p_xx), ("p_y", p_y), ("p_yy", p_yy),
            ("u u_x", u * u_x), ("v u_y", v * u_y),
            ("(u_xx + u_yy)", u_lap), ("(v_xx + v_yy)", v_lap),
        ])
    return deduplicate([
        ("1", torch.ones_like(u)), ("u", u), ("v", v),
        ("p_x", p_x), ("p_xx", p_xx), ("p_y", p_y), ("p_yy", p_yy),
        ("u v_x", u * v_x), ("v v_y", v * v_y),
        ("(u_xx + u_yy)", u_lap), ("(v_xx + v_yy)", v_lap),
    ])


class ConfiguredLibrary(Library):
    def __init__(self, filename, params):
        super().__init__()
        self.filename = filename
        self.params = params
        self.axis_names = axis_names_for_mode(params["data_mode"])
        self.coord_index = axis_to_coord_index_for_mode(params["data_mode"])
        lib_config = params["sindy_config"].get("library", {})
        self.variable_names = lib_config.get(
            "variable_names",
            ["u"] if params["output_dim"] == 1 else [f"x{idx}" for idx in range(params["output_dim"])],
        )
        self.targets = params["sindy_config"].get("targets", default_targets(self.variable_names))
        self.target_names = [target_name(target, self.axis_names) for target in self.targets]
        self.feature_names = [[] for _ in self.targets]
        self.fixed_values = params["fixed_library"]["values"]
        self.coord_lookup = params["fixed_library"]["coord_lookup"]

    def sample_indices(self, coords):
        rows = coords.detach().cpu().numpy()
        indexes = []
        for row in rows:
            key = coord_key(row)
            if key not in self.coord_lookup:
                raise KeyError(f"Coordinate {row.tolist()} is not in the fixed derivative grid")
            indexes.append(self.coord_lookup[key])
        return torch.as_tensor(indexes, dtype=torch.long, device=coords.device)

    def library(self, input):
        prediction, coords = input
        state = LibraryState(
            coords=coords,
            variable_names=self.variable_names,
            axis_names=self.axis_names,
            coord_index=self.coord_index,
            fixed_values=self.fixed_values,
            sample_indices=self.sample_indices(coords),
        )
        lib_config = self.params["sindy_config"].get("library", {})

        time_derivs = []
        thetas = []
        feature_names = []
        for target, lhs_name in zip(self.targets, self.target_names):
            time_derivs.append(state.deriv(target["variable"], target.get("axis", "t"), target.get("order", 1)))
            if "feature_tokens" in target:
                specs = [(name, parse_derivative_token(name, state)) for name in target["feature_tokens"]]
            elif lib_config.get("type") == "navier_stokes":
                specs = ns_specs(state, target["variable"])
            else:
                specs = configured_specs(state, self.params, self.axis_names)
            specs = [(name, value) for name, value in specs if not uses_target(name, lhs_name)]
            feature_names.append([name for name, _ in specs])
            thetas.append(torch.cat([value for _, value in specs], dim=1))

        self.feature_names = feature_names
        return time_derivs, thetas


def save_combined_results(results):
    output_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as handle:
        json.dump([results], handle, indent=2)


def crop_widths_for_shape(shape, crop):
    if not crop:
        return (0,) * len(shape)
    if isinstance(crop, int):
        return (crop,) * len(shape)
    return tuple(int(width) for width in crop)


def crop_axis(values, width):
    if values is None or width <= 0:
        return values
    return values[width:-width]


def crop_array(array, crop):
    widths = crop_widths_for_shape(array.shape, crop)
    return crop_array_with_widths(array, widths)


def crop_array_with_widths(array, widths):
    slices = []
    for size, width in zip(array.shape, widths):
        if width <= 0:
            slices.append(slice(None))
        else:
            slices.append(slice(width, size - width))
    return array[tuple(slices)], widths


def crop_values_with_widths(array, widths):
    cropped, _ = crop_array_with_widths(np.asarray(array, dtype=float), widths)
    return cropped


def coord_key(row, decimals=7):
    return tuple(np.round(np.asarray(row, dtype=float), decimals=decimals))


def default_variable_names_for_arrays(arrays):
    return [f"u{index}" if len(arrays) > 1 else "u" for index in range(len(arrays))]


def configured_max_deriv_order(data_shape, params):
    lib_config = params["sindy_config"].get("library", {})
    return tuple(lib_config.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"])[:len(data_shape)])


def build_fixed_custom_values(fixed_values, bundle, data_shape, widths, x, t, params):
    token_names = params["sindy_config"].get("library", {}).get("custom_tokens", [])
    if "d_x(u u_x)" not in token_names:
        return
    if "u" not in bundle["variables"] or "x" not in bundle["axis_names"]:
        return

    u = bundle["variables"]["u"]["values"]
    axis_index = bundle["axis_names"].index("x")
    orders = [0] * len(bundle["axis_names"])
    orders[axis_index] = 1
    u_x = bundle["variables"]["u"]["derivatives"][tuple(orders)]
    fixed_values["d_x(u u_x)"] = crop_values_with_widths(
        numpy_gradient_derivative(u * u_x, x[1] - x[0], axis=axis_index, order=1),
        widths,
    ).reshape(-1, 1)


def build_fixed_library_data(coords, data_arrays, x, y, z, t, widths, params):
    lib_config = params["sindy_config"].get("library", {})
    variable_names = lib_config.get("variable_names", default_variable_names_for_arrays(data_arrays))
    bundle = compute_derivative_bundle(
        data_arrays if len(data_arrays) > 1 else data_arrays[0],
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=variable_names,
        max_orders=configured_max_deriv_order(data_arrays[0].shape, params),
    )

    fixed_values = {}
    for variable_name in variable_names:
        variable = bundle["variables"][variable_name]
        for orders, values in variable["derivatives"].items():
            name = derivative_name(variable_name, orders, bundle["axis_names"])
            fixed_values[name] = crop_values_with_widths(values, widths).reshape(-1, 1)

    build_fixed_custom_values(fixed_values, bundle, data_arrays[0].shape, widths, x, t, params)

    coords_for_lookup = torch.from_numpy(coords.reshape(-1, coords.shape[-1])).float().numpy()
    coord_lookup = {
        coord_key(row): index
        for index, row in enumerate(coords_for_lookup)
    }
    return {
        "values": {
            name: torch.from_numpy(values).float()
            for name, values in fixed_values.items()
        },
        "coord_lookup": coord_lookup,
    }


def prepare_tensors(data, x, y, z, t, params):
    mode = params["data_mode"]
    crop = params.get("crop", 0)

    if mode == "ode_scalar":
        raw_values = np.asarray(data[0] if isinstance(data, list) else data, dtype=float).reshape(-1)
        data_arrays = [raw_values]
        values = raw_values
        width = crop_widths_for_shape(values.shape, crop)[0]
        values = values[width:-width] if width else values
        t_values = crop_axis(np.asarray(t, dtype=float), width)
        coords = t_values[:, None]
        target = values[:, None]
        widths = (width,)

    elif mode == "ode_system":
        data_arrays = [np.asarray(values, dtype=float) for values in data]
        width = crop_widths_for_shape(data_arrays[0].shape, crop)[0]
        t_values = crop_axis(np.asarray(t, dtype=float), width)
        coords = t_values[:, None]
        target = np.column_stack([
            np.asarray(values, dtype=float)[width:-width] if width else np.asarray(values, dtype=float)
            for values in data_arrays
        ])
        widths = (width,)

    elif mode == "pde_1d":
        data_arrays = [np.asarray(data, dtype=float)]
        values, widths = crop_array(data_arrays[0], crop)
        t_values = crop_axis(np.asarray(t, dtype=float), widths[0])
        x_values = crop_axis(np.asarray(x, dtype=float), widths[1])
        t_grid, x_grid = np.meshgrid(t_values, x_values, indexing="ij")
        coords = np.stack((t_grid, x_grid), axis=-1)
        target = values[..., None]

    elif mode == "pde_2d_system":
        data_arrays = [np.asarray(values, dtype=float) for values in data]
        cropped, widths = crop_array(data_arrays[0], crop)
        u = cropped
        v = data_arrays[1][tuple(slice(width, size - width) if width else slice(None) for size, width in zip(data_arrays[1].shape, widths))]
        p = data_arrays[2][tuple(slice(width, size - width) if width else slice(None) for size, width in zip(data_arrays[2].shape, widths))]
        t_values = crop_axis(np.asarray(t, dtype=float), widths[0])
        y_values = crop_axis(np.asarray(y, dtype=float), widths[1])
        x_values = crop_axis(np.asarray(x, dtype=float), widths[2])
        t_grid, y_grid, x_grid = np.meshgrid(t_values, y_values, x_values, indexing="ij")
        coords = np.stack((t_grid, x_grid, y_grid), axis=-1)
        target = np.stack((u, v, p), axis=-1)

    else:
        raise ValueError(f"Unknown DeepMoD data mode: {mode}")

    fixed_library = build_fixed_library_data(coords, data_arrays, x, y, z, t, widths, params)
    return torch.from_numpy(coords).float(), torch.from_numpy(target).float(), fixed_library


def build_dataset(coords, target, params, device):
    def load_fn():
        return coords, target

    return Dataset(
        load_fn,
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": params["number_of_samples"]},
        preprocess_kwargs={
            "random_state": params.get("seed", 0),
            "noise_level": params.get("noise_level", 0.0),
            "normalize_coords": False,
            "normalize_data": False,
        },
        device=device,
    )


def build_library(params):
    return ConfiguredLibrary(params["filename"], params)


def build_estimator(params):
    estimator = params.get("sparse_estimator", "threshold")
    if estimator == "pdefind":
        return PDEFIND(
            lam=params.get("pdefind_lam", 1e-3),
            dtol=params.get("pdefind_dtol", params["threshold"]),
        )
    return Threshold(params["threshold"])


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
    params["coefficient_tol"] = params.get(
        "coefficient_tol",
        sindy_params[filename].get("optimizer", {}).get("coefficient_tol", 0.0),
    )
    return params


def build_model(params, device):
    if params.get("network", "tanh") == "siren":
        network = Siren(
            params["input_dim"],
            params["hidden_layers"],
            params["output_dim"],
            first_omega_0=params.get("first_omega_0", 30.0),
            hidden_omega_0=params.get("hidden_omega_0", 30.0),
        )
    else:
        network = NN(params["input_dim"], params["hidden_layers"], params["output_dim"])

    library = build_library(params)
    model = DeepMoD(network, library, build_estimator(params), LeastSquares()).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=params.get("betas", (0.99, 0.99)),
        amsgrad=True,
        lr=params["learning_rate"],
    )
    scheduler = TrainTestPeriodic(
        periodicity=params["scheduler_periodicity"],
        patience=params["scheduler_patience"],
        delta=params["scheduler_delta"],
    )
    return model, optimizer, scheduler, library


def pretrain_network(model, train_dataloader, optimizer, iterations, write_iterations):
    if iterations <= 0:
        return
    print(f"Pretraining NN for {iterations} iterations")
    for iteration in range(iterations):
        losses = []
        for coords, target in train_dataloader:
            prediction = model.func_approx(coords)[0]
            loss = torch.mean((prediction - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        if iteration == 0 or (iteration + 1) % write_iterations == 0:
            print(f"  pretrain {iteration + 1:5d}  MSE: {torch.stack(losses).mean().item():.2e}")


def fit_sparse_once(model, train_dataloader):
    for coords, _ in train_dataloader:
        _, time_derivs, thetas = model(coords)
        model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
        _ = model(coords)
        return
    raise ValueError("Cannot fit sparse coefficients: empty train dataloader")


def collect_result(filename, library, model, params):
    coefficients = [
        tensor.detach().cpu().numpy().reshape(-1)
        for tensor in model.constraint_coeffs(scaled=False, sparse=True)
    ]
    coefficient_tol = params.get("coefficient_tol", 0.0)
    if coefficient_tol > 0:
        coefficients = [
            np.where(np.abs(values) < coefficient_tol, 0.0, values)
            for values in coefficients
        ]

    for target_name, feature_names, values in zip(library.target_names, library.feature_names, coefficients):
        print(f"{target_name} = {compact_terms(feature_names, values)}")

    library_sizes = {
        target_name: len(feature_names)
        for target_name, feature_names in zip(library.target_names, library.feature_names)
    }
    return {
        "dataset": filename.split(".")[0],
        "targets": library.target_names,
        "features": library.feature_names,
        "coefficients": [values.tolist() for values in coefficients],
        "library_sizes": library_sizes,
        "library_size": sum(library_sizes.values()),
    }


def run_deepmod(data, x, y, z, t, filename):
    params = build_run_params(filename)
    set_seeds(params.get("seed", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords, target, fixed_library = prepare_tensors(data, x, y, z, t, params)
    params["fixed_library"] = fixed_library
    dataset = build_dataset(coords, target, params, device)
    train_dataloader, test_dataloader = get_train_test_loader(
        dataset,
        train_test_split=params["train_test_split"],
    )
    model, optimizer, scheduler, library = build_model(params, device)
    pretrain_network(
        model,
        train_dataloader,
        optimizer,
        iterations=params.get("pretrain_iterations", 0),
        write_iterations=params.get("pretrain_write_iterations", params["write_iterations"]),
    )

    if params.get("train_after_pretrain", True):
        train(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            log_dir=params.get("log_dir", f"deepmod_runs/{filename.split('.')[0]}"),
            split=params["train_test_split"],
            max_iterations=params["max_iterations"],
            write_iterations=params["write_iterations"],
            patience=params["convergence_patience"],
            delta=params["convergence_delta"],
        )
        print()
    else:
        fit_sparse_once(model, train_dataloader)

    return collect_result(filename, library, model, params)


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
