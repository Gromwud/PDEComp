import numpy as np
from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import torch
from pathlib import Path
import scipy.io as scio
import json
import time
from datetime import datetime
import pandas as pd

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/epde")
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

    "ODE_simple_discovery"
]


def save_combined_results(results):
    """Save results to a common JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
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
        y = None
        z = None

    elif filename == "kdv_periodic_data.npy":
        data = np.load(DATA_DIR / filename)
        shape = len(data)
        t = np.linspace(0, 1, shape)
        x = np.linspace(0, 1, shape)
        y = None
        z = None

    elif filename == "ac_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0.0, 1.0, 51)
        x = np.linspace(-1.0, 0.984375, 128)
        y = None
        z = None

    elif filename == "kdv_data.mat" or filename == "burgers_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        data = np.real(system["usol"]).T
        t = np.ravel(system["t"])
        x = np.ravel(system["x"])
        y = None
        z = None

    elif filename == "pde_divide_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 1, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "pde_compound_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 0.5, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "wave_data.csv":
        data = np.loadtxt(DATA_DIR / filename, delimiter=',').T
        t = np.linspace(0, 1, 81)
        x = np.linspace(0, 1, 81)
        y = None
        z = None

    elif filename == "lorenz_data.npy":
        system = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lorenz_time.npy")
        end = 1000
        data = [system[:end, 0], system[:end, 1], system[:end, 2]]
        t = t[:end]
        x = None
        y = None
        z = None

    elif filename == "lotka_data.npy":
        system = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lotka_time.npy")
        end = 150
        t = t[:end]
        data = [system[:end, 0], system[:end, 1]]
        x = None
        y = None
        z = None
    
    elif filename == "ns_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        U_star = system['U_star']  # N x 2 x T
        P_star = system['p_star']  # N x T
        t_star = system['t']  # T x 1
        X_star = system['X_star']  # N x 2

        t_train = 50

        t = t_star.flatten()  # N x T
        x = np.unique(X_star[:, 0:1].flatten())  # N x T
        y = np.unique(X_star[:, 1:2].flatten()) # N x T
        z = None

        u = U_star[:, 0, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
        v = U_star[:, 1, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
        p = P_star.T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]   # N x T

        data = [u, v, p]

    elif filename == "ks_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        data = system['uu'].T
        t = np.ravel(system['tt'])
        x = np.ravel(system['x'])
        y = None
        z = None
    
    elif filename == "burgers_sln_100_data.csv":
        df = pd.read_csv(DATA_DIR / filename, header=None)

        u = df.values
        data = np.transpose(u)
        t = np.linspace(0, 1, 101)
        x = np.linspace(-1000, 0, 101)
        y = None
        z = None
    
    elif filename == "ODE_simple_discovery":
        C = 1.3
        t = np.linspace(0, 4 * np.pi, 200)
        data = np.sin(t) + C * np.cos(t)
        data = [data, ]
        x = None
        y = None
        z = None

    return data, x, y, z, t


def run_epde(data, x, y, z, t, filename):
    """Основная логика идентификации"""
    start = time.perf_counter()
    
    if filename == "ac_data.npy":
        grid = np.meshgrid(t, x, indexing="ij")
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

    elif filename == "ode_data.npy":
        dimensionality = 0
        trig_tokens = TrigonometricTokens(
            freq=(2 - 1e-8, 2 + 1e-8), dimensionality=dimensionality
        )
        grid_tokens = GridTokens(
            [
                "x_0",
            ],
            max_power=2,
            dimensionality=dimensionality,
        )

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=[
                t,
            ],
            verbose_params={"show_iter_idx": True},
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        epde_search_obj.fit(
            data=[
                data,
            ],
            variable_names=[
                "u",
            ],
            max_deriv_order=(2, 2),
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[trig_tokens, grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-12, 1e-4),
        )

    elif filename == "kdv_data.mat":
        grid = np.meshgrid(t, x, indexing="ij")
        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=grid,
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8

        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        bounds = (1e-5, 1e-2)
        epde_search_obj.fit(
            data=data,
            variable_names=[
                "u",
            ],
            max_deriv_order=(1, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False,
        )

    elif filename == "kdv_periodic_data.npy":
        dimensionality = data.ndim - 1
        grid = np.meshgrid(t, x, indexing="ij")
        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=grid,
            device="cuda",
        )   
        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )
        popsize = 12

        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

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

        trig_tokens = TrigonometricTokens(
            dimensionality=dimensionality, freq=(0.999, 1.001)
        )
        trig_tokens._token_family.set_status(
            unique_specific_token=True, unique_token_type=False, meaningful=True
        )

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        bounds = (1e-10, 1e-2)
        epde_search_obj.fit(
            data=data,
            variable_names=[
                "u",
            ],
            max_deriv_order=(1, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[custom_trig_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False,
        )

    elif filename == "burgers_data.mat":
        grid = np.meshgrid(t, x, indexing="ij")
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

        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        bounds = (1e-4, 1e-0)

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
        )

    elif filename == "vdp_data.npy":
        dimensionality = 0

        trig_tokens = TrigonometricTokens(
            freq=(2 - 1e-8, 2 + 1e-8), dimensionality=dimensionality
        )
        grid_tokens = GridTokens(
            [
                "x_0",
            ],
            dimensionality=dimensionality,
            max_power=2,
        )

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=(t,),
            verbose_params={"show_iter_idx": True},
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        epde_search_obj.fit(
            data=[
                data,
            ],
            variable_names=[
                "u",
            ],
            max_deriv_order=(2, 2),
            equation_terms_max_number=5,
            data_fun_pow=2,
            additional_tokens=[trig_tokens, grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-8, 1e-0),
        )

    elif filename == "pde_divide_data.npy":
        grid = np.meshgrid(t, x, indexing="ij")
        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=20,
            coordinate_tensors=grid,
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=50)

        grid_tokens = CacheStoredTokens(
            token_type="grid",
            token_labels=["t", "x"],
            token_tensors={"t": grid[0], "x": grid[1]},
            params_ranges={"power": (1, 1)},
            params_equality_ranges=None,
        )

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}
        bounds = (1e-9, 1e-2)

        epde_search_obj.fit(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False,
        )

    elif filename == "pde_compound_data.npy":
        grid = np.meshgrid(t, x, indexing="ij")
        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=20,
            coordinate_tensors=grid,
            device="cuda",
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=50)

        grid_tokens = CacheStoredTokens(
            token_type="grid",
            token_labels=["t", "x"],
            token_tensors={"t": grid[0], "x": grid[1]},
            params_ranges={"power": (1, 1)},
            params_equality_ranges=None,
        )

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}
        bounds = (1e-9, 1e-2)

        epde_search_obj.fit(
            data=data,
            variable_names=["u"],
            max_deriv_order=(2, 3),
            derivs=None,
            equation_terms_max_number=5,
            data_fun_pow=1,
            additional_tokens=[grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False,
        )

    elif filename == "wave_data.csv":
        grid = np.meshgrid(t, x, indexing="ij")
        epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=(grid[..., 0], grid[..., 1]), device='cuda')

        # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
        #                                     preprocessor_kwargs={'epochs_max' : 1e3})
        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                        preprocessor_kwargs={})
        popsize = 8

        epde_search_obj.set_moeadd_params(population_size=popsize,
                                        training_epochs=5)
        
        factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

        bounds = (1e-12, 1e-2)
        epde_search_obj.fit(data=data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                            equation_terms_max_number=5, data_fun_pow=3,
                            additional_tokens=[],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=bounds, fourier_layers=False)

    elif filename == "lorenz_data.npy":
        dimensionality = data[0].ndim - 1

        trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                        dimensionality=dimensionality)
        grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

        epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                    coordinate_tensors=[t, ],
                                    device='cuda')

        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                        preprocessor_kwargs={})

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=40)

        factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

        epde_search_obj.fit(data=data, variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                            equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-4, 1e-0))

    elif filename == "lotka_data.npy":
        dimensionality = data[0].ndim - 1

        trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                        dimensionality=dimensionality)
        grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

        epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                    coordinate_tensors=(t,),
                                    device='cuda')

        epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                        preprocessor_kwargs={})

        popsize = 8
        epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=30)

        factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

        epde_search_obj.fit(data=data, variable_names=['u', 'v'], max_deriv_order=(1,),
                            equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[trig_tokens, ],
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-4, 1e-0))
    
    elif filename == "ks_data.mat":
        grid = np.meshgrid(t, x, indexing="ij")

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=5,
            coordinate_tensors=grid,
            device='cuda'
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type="FD", preprocessor_kwargs={}
        )

        popsize = 16

        epde_search_obj.set_moeadd_params(population_size=popsize,
                                          training_epochs=10)

        factors_max_number = {"factors_num": [1, 2], "probas": [0.65, 0.35]}

        bounds = (1e-12, 1e-0)
        epde_search_obj.fit(
            data=data,
            variable_names=["u"],
            max_deriv_order=(1, 4),
            derivs=None,
            equation_terms_max_number=10,
            data_fun_pow=1,
            additional_tokens=[],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False
        )

    elif filename == "ns_data.mat":
        t_train = 50
        grid = np.meshgrid(t[:t_train], y, x, indexing = 'ij')

        epde_search_obj = EpdeSearch(
            use_solver=False,
            multiobjective_mode=True,
            use_pic=True,
            boundary=5,
            coordinate_tensors=grid,
            device='cuda'
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type='FD', preprocessor_kwargs={}
        )

        popsize = 16

        epde_search_obj.set_moeadd_params(population_size=popsize,
                                          training_epochs=50)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

        bounds = (1e-12, 1e-0)
        epde_search_obj.fit(
            data=data,
            variable_names=["u", "v", "p"],
            max_deriv_order=(1, 2, 2),
            derivs=None,
            equation_terms_max_number=10,
            data_fun_pow=1,
            additional_tokens=[],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=bounds,
            fourier_layers=False
        )

    elif filename == "burgers_sln_100_data.csv":
        grid = np.meshgrid(t, x, indexing = 'ij')

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=(10, 10),
            coordinate_tensors=grid,
            device='cuda'
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type='poly', preprocessor_kwargs={}
        )

        popsize = 16

        epde_search_obj.set_moeadd_params(population_size=popsize,
                                          training_epochs=5)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

        epde_search_obj.fit(
            data=data,
            variable_names=['u', ],
            max_deriv_order=(2, 3),
            equation_terms_max_number=5,
            data_fun_pow=3,
            additional_tokens=None,
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-6, 1e-0)
        )
    
    elif filename == "ODE_simple_discovery":
        dimensionality = data[0].ndim - 1

        trig_tokens = TrigonometricTokens(freq = (0.999, 1.001),
                                        dimensionality=dimensionality)
        
        grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

        epde_search_obj = EpdeSearch(
            use_solver=False,
            use_pic=True,
            boundary=10,
            coordinate_tensors=[t,],
            verbose_params={'show_iter_idx': True},
            device='cuda'
        )

        epde_search_obj.set_preprocessor(
            default_preprocessor_type='FD', preprocessor_kwargs={}
        )

        popsize = 8

        epde_search_obj.set_moeadd_params(population_size=popsize,
                                          training_epochs=20)

        factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

        epde_search_obj.fit(
            data=data,
            variable_names=['u', ],
            max_deriv_order=(2, 3),
            equation_terms_max_number=5,
            data_fun_pow=3,
            additional_tokens=[trig_tokens, grid_tokens],
            equation_factors_max_number=factors_max_number,
            eq_sparsity_interval=(1e-4, 1e-0)
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
