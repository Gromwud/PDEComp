import os
import numpy as np
from epde.interface.interface import EpdeSearch
from epde import TrigonometricTokens
from pathlib import Path
import scipy.io as scio
import json

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/epde")
DATASETS = [
    "ac_data.npy",
    # "kdv_data.mat",
    # "burgers_data.mat",
    # "vdp_data.npy",
    # "pde_divide_data.npy"
]


def save_combined_results(results):
    """Save results to a common JSON file"""
    output_file = RESULTS_DIR / "results.json"
    output_file.parent.mkdir(exist_ok=True)

    result = []

    result.append(results)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def load_data(filename):
    """Загрузка данных без привязки к структуре базового класса"""

    if filename == "ode_data.npy" or filename == "vdp_data.npy":
        data = np.load(DATA_DIR / filename)
        step = 0.05
        steps_num = 320
        t = np.arange(start=0.0, stop=step * steps_num, step=step)
        x = None

    elif filename == "kdv_data_periodic.npy":
        data = np.load(DATA_DIR / filename)
        shape = len(data)
        t = np.linspace(0, 1, shape)
        x = np.linspace(0, 1, shape)

    elif filename == "ac_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0.0, 1.0, 51)
        x = np.linspace(-1.0, 0.984375, 128)

    elif filename == "kdv_data.mat" or filename == "burgers_data.mat":
        data = scio.loadmat(DATA_DIR / filename)
        t = np.ravel(data["t"])
        x = np.ravel(data["x"])
        data = np.real(data["usol"]).T

    elif filename == "pde_divide_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 0.5, 251)
        x = np.linspace(1, 2, 100)

    return data, x, t


def run_epde(data, x, t, filename):
    """Основная логика идентификации"""
    grid = np.meshgrid(t, x, indexing="ij")

    if filename == "ac_data.npy":
        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=20,
            coordinate_tensors=grid,
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8

        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=30)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        bounds = (1e-5, 1e-2)
        epde_search_obj.fit(
            data=data,
            variable_names=[
                "u",
            ],
            max_deriv_order=(2, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=3,
            additional_tokens=[],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False,
        )  # , data_nn=data_nn

    elif filename == "kdv_data.mat":
        ...

    elif filename == "burgers_data.mat":
        ...

    elif filename == "vdp_data.npy":
        ...

    elif filename == "pde_divide_data.npy":
        ...

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    result = {
        "dataset": filename.split(".")[0],
        "coefficients": [],
        "features": [],
        # "model_str": str(model.print(precision=4))
    }

    return result


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, t = load_data(dataset)
            result = run_epde(data, x, t, dataset)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
