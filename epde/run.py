import numpy as np
import sys
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import torch
from pathlib import Path
import json
import time
from datetime import datetime

from data.dataloader import load_data
from data.config import epde_params, COMMON_PARAMS
from data.derivatives import build_epde_derivatives

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/epde")
DATASETS = [
    # "ode_data.npy",
    # "vdp_data.npy",

    # "lorenz_data.npy",
    # "lotka_data.npy",

    # "burgers_data.mat",
    # "ac_data.npy",
    # "kdv_data.mat",
    # "kdv_periodic_data.npy",
    # "wave_data.csv",
    # "pde_divide_data.npy",
    "pde_compound_data.npy",
    # "ns_data.mat",
    # "ks_data.mat",
    
    # "burgers_sln_100_data.csv",
    
    # "ODE_simple_discovery"
]


def save_combined_results(results):
    """Save all dataset results into one JSON file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    result = []

    result.append(results)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def get_coordinate_tensors(coordinate_tensors, t, x, y, z):
    """Build coordinate tensors in the format expected by EPDE."""

    if coordinate_tensors == None:
        # 2d
        return np.meshgrid(t, x, indexing = 'ij')

    elif coordinate_tensors == '1d':
        return [t, ]

    elif coordinate_tensors == '3d':
        return np.meshgrid(t, y, x, indexing = 'ij')


def get_additional_tokens(additional_tokens, grid, data, trig_tokens_freq):
    """Create optional EPDE token families configured for a dataset."""

    if additional_tokens == None:
        return []

    elif additional_tokens == 'CacheStoredTokens':
        grid_tokens = CacheStoredTokens(
            token_type="grid",
            token_labels=["t", "x"],
            token_tensors={"t": grid[0], "x": grid[1]},
            params_ranges={"power": (1, 1)},
            params_equality_ranges=None,
        )
        return [grid_tokens]

    elif additional_tokens == 'TrigonometricTokens':
        dimensionality = data[0].ndim - 1

        trig_tokens = TrigonometricTokens(freq=trig_tokens_freq,
                                          dimensionality=dimensionality)
        return [trig_tokens]

    elif additional_tokens == 'TrigonometricTokens, GridTokens':
        dimensionality = 0
        trig_tokens = TrigonometricTokens(
            freq=trig_tokens_freq, dimensionality=dimensionality
        )
        grid_tokens = GridTokens(
            [
                "x_0",
            ],
            dimensionality=dimensionality,
            max_power=2,
        )
        return [trig_tokens, grid_tokens]
    
    elif additional_tokens == 'ODE_simple_discovery':
        dimensionality = data[0].ndim - 1
        trig_tokens = TrigonometricTokens(
            freq=trig_tokens_freq, dimensionality=dimensionality
        )
        grid_tokens = GridTokens(
            [
                "x_0",
            ],
            dimensionality=dimensionality,
            max_power=2,
        )
        return [trig_tokens, grid_tokens]

    elif additional_tokens == 'custom_trig_tokens':
        custom_trigonometric_eval_fun = {
            "cos(t)sin(x)": lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1]))
            ** kwargs["power"]
        }
        custom_trig_evaluator = CustomEvaluator(
            custom_trigonometric_eval_fun, eval_fun_params_labels=["power"]
        )
        trig_params_ranges = {"power": (1, 1)}
        trig_params_equal_ranges = {}
        custom_trig_tokens = CustomTokens(
            token_type="trigonometric",
            token_labels=["cos(t)sin(x)"],
            evaluator=custom_trig_evaluator,
            params_ranges=trig_params_ranges,
            params_equality_ranges=trig_params_equal_ranges,
            meaningful=True,
            unique_token_type=False,
        )
        return [custom_trig_tokens]


def normalize_epde_max_deriv_order(max_deriv_order, data):
    """Interpret (time, spatial) derivative limits for any dataset dimensionality."""

    ndim = np.asarray(data).ndim
    if isinstance(max_deriv_order, int):
        return tuple([max_deriv_order] * ndim)

    orders = tuple(max_deriv_order)
    if len(orders) == ndim:
        return orders
    if len(orders) > ndim:
        return orders[:ndim]
    if len(orders) == 2 and ndim > 2:
        return (orders[0],) + tuple([orders[1]] * (ndim - 1))

    raise ValueError(f"Expected derivative order for {ndim} axes, got {max_deriv_order}")


def run_epde(data, x, y, z, t, filename):
    """Run EPDE discovery for one configured dataset."""

    start = time.perf_counter()

    params = epde_params[filename]
    epde_data = data if isinstance(data, list) else [data]
    
    use_solver = params["use_solver"]
    use_pic = params["use_pic"]
    bounds = params["boundary"]
    multiobjective_mode = params.get("multiobjective_mode", True)

    population_size = params["population_size"]
    training_epochs = params["training_epochs"]

    max_deriv_order = normalize_epde_max_deriv_order(params.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]), epde_data[0])
    derivs = params.get("derivs", None)
    equation_terms_max_number = params["equation_terms_max_number"]
    data_fun_pow = params.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"])
    additional_tokens = params.get("additional_tokens", None)
    equation_factors_max_number = params["equation_factors_max_number"]
    eq_sparsity_interval = params["eq_sparsity_interval"]
    default_preprocessor_type = params["default_preprocessor_type"]
    variable_names = params["variable_names"]
    fourier_layers = params.get("fourier_layers", False)
    coordinate_tensors = params.get("coordinate_tensors", None)
    trig_tokens_freq = params.get("trig_tokens_freq", None)


    grid = get_coordinate_tensors(coordinate_tensors, t, x, y, z)        
    if derivs is None:
        derivs = build_epde_derivatives(
            data=epde_data,
            x=x,
            y=y,
            z=z,
            t=t,
            variable_names=variable_names,
            max_deriv_order=max_deriv_order,
        )

    epde_search_obj = EpdeSearch(
        use_solver=use_solver,
        multiobjective_mode=multiobjective_mode,
        use_pic=use_pic,
        boundary=bounds,
        coordinate_tensors=grid,
        device='cuda'
    )

    epde_search_obj.set_preprocessor(
        default_preprocessor_type=default_preprocessor_type, preprocessor_kwargs={}
    )

    epde_search_obj.set_moeadd_params(population_size=population_size,
                                        training_epochs=training_epochs)

    epde_search_obj.fit(
        data=epde_data,
        variable_names=variable_names,
        max_deriv_order=max_deriv_order,
        derivs=derivs,
        equation_terms_max_number=equation_terms_max_number,
        data_fun_pow=data_fun_pow,
        additional_tokens=get_additional_tokens(additional_tokens, grid, epde_data, trig_tokens_freq),
        equation_factors_max_number=equation_factors_max_number,
        eq_sparsity_interval=eq_sparsity_interval,
        fourier_layers=fourier_layers
    )


    epde_search_obj.equations(only_print=True, num=1)
    finish = time.perf_counter()
    elapsed_time = finish - start
    epde_search_obj.visualize_solutions()

    result = {
        "dataset": filename.split(".")[0],
        "coefficients": [],
        "features": [],
        "time": elapsed_time,
        # "model_str": str(model.print(precision=4))
    }

    return result


if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            result = run_epde(data, x, y, z, t, dataset)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
