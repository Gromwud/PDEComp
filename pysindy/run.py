import numpy as np
import pysindy as ps
from pathlib import Path
import scipy.io as scio
import json
from datetime import datetime

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/pysindy")
DATASETS = [
    "ac_data.npy",
    "kdv_data.mat",
    "burgers_data.mat",
    # "vdp_data.npy",
    "pde_divide_data.npy"
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
        t = np.linspace(0, 1, shape + 1)
        x = np.linspace(0, 1, shape + 1)
        y = None
        z = None

    elif filename == "lorenz_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lorenz_time.npy")
        end = 1000
        t = t[:end]
        x = data[:end, 0]
        y = data[:end, 1]
        z = data[:end, 2]

    elif filename == "lotka_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lotka_time.npy")
        end = 150
        t = t[:end]
        x = data[:end, 0]
        y = data[:end, 1]
        z = None

    if len(data.shape) == 1:
        data = data.T.reshape(len(t), 1)
    elif len(data.shape) == 2:
        data = data.T.reshape(len(x), len(t), 1)

    return data, x, t


def run_sindy(data, x, t, filename):
    """Основная логика идентификации"""
    if filename == "ac_data.npy":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=3, include_bias=False),
            # function_names=library_function_names,
            derivative_order=3,
            spatial_grid=x,
            # temporal_grid=t,
            include_bias=True,
        ).fit(data)
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)

    elif filename == "kdv_data.mat":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)

    elif filename == "burgers_data.mat":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)

    elif filename == "vdp_data.npy":
        library_functions = [lambda x: x, lambda x: x * x]
        library_function_names = [lambda x: x, lambda x: x + x]

        library = ps.SINDyPILibrary(
            library_functions=library_functions,
            # x_dot_library_functions=x_dot_library_functions,
            t=t,
            include_bias=True,
            # implicit_terms=True,
            # derivative_order=2
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.SINDyPI(
            threshold=1e-4,
            tol=1e-5,
            thresholder="l1",
            max_iter=6000,
            # normalize_columns=True
        )

    elif filename == "pde_divide_data.npy":
        functions = [
            lambda x : 1/x,
            lambda x : 1/x,
                     ]
        functions_names = [lambda x : "1/" + x]
        
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=2,
            spatial_grid=x,
            include_bias=True,
        )

        lib_custom = ps.CustomLibrary(library_functions=functions, function_names=functions_names) * library

        library = ps.ConcatLibrary([lib_custom, library]).fit(data)
        print(library.get_feature_names(), "\n")

        # optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=False)
        optimizer = ps.SR3(tol=1e-15, normalize_columns=True, max_iter=10000)
        # optimizer = ps.SR3(threshold=5, max_iter=10000, tol=1e-15, thresholder='l1', normalize_columns=True)
        # optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
        # optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)

    elif filename == "lorenz_data.npy":
        ...

    elif filename == "lotka_data.npy":
        ...

    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(data, t=t[1] - t[0])
    model.print(precision=4)

    result = {
        "dataset": filename.split(".")[0],
        "coefficients": model.coefficients().tolist(),
        "features": model.get_feature_names(),
        # "model_str": str(model.print(precision=4))
    }

    return result


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, t = load_data(dataset)
            result = run_sindy(data, x, t, dataset)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
