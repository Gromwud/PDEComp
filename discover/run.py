import os
import numpy as np
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='numpy.*')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.*')

# from dso.PDE_discover import SymEqOptimizer, DeepSymbolicOptimizer
from dso import DeepSymbolicOptimizer_PDE
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


def run_discover(data, x, t, filename):
    """Основная логика идентификации"""
    grid = np.meshgrid(t, x, indexing="ij")

    if filename == "ac_data.npy":
        config_file_path = "./dso/config/MODE1/config_pde_KdV.json"

    elif filename == "kdv_data.mat":
        ...

    elif filename == "burgers_data.mat":
        ...

    elif filename == "vdp_data.npy":
        ...

    elif filename == "pde_divide_data.npy":
        ...

    model = DeepSymbolicOptimizer_PDE(config_file_path)

    result = model.train()

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
            result = run_discover(data, x, t, dataset)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
