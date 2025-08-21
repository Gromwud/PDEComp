import os
import numpy as np

from codes import *
from codes.sga import SGA
from pathlib import Path
import scipy.io as scio
import json
import torch
from datetime import datetime

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
        data = scio.loadmat(DATA_DIR / filename)
        data = np.real(data["usol"]).T
        t = np.ravel(data["t"])
        x = np.ravel(data["x"])
        y = None
        z = None

    elif filename == "pde_divide_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 1, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "pde_compound_data.npy":
        data = np.load(filename)
        t = np.linspace(0, 0.5, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "wave_data.npy":
        data = np.loadtxt(filename, delimiter=',').T
        t = np.linspace(0, 1, shape + 1)
        x = np.linspace(0, 1, shape + 1)
        y = None
        z = None

    elif filename == "lorenz_data.npy":
        data = np.load(filename)
        t = np.load("lorenz_time.npy")
        end = 1000
        t = t[:end]
        x = data[:end, 0]
        y = data[:end, 1]
        z = data[:end, 2]

    elif filename == "lotka_data.npy":
        data = np.load(filename)
        t = np.load("lotka_time.npy")
        end = 150
        t = t[:end]
        x = data[:end, 0]
        y = data[:end, 1]
        z = None
                
    return data, x, y, z, t


def run_discover(filename):
    """Основная логика идентификации"""
    if filename == "ac_data.npy":
        config_file_path = "/tmp/discover/dso/dso/config/MODE1/config_pde_KdV.json"

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
    # print("CUDA available: ", torch.cuda.is_available())
    # all_results = []
    # for dataset in DATASETS:
    #     print(f"\n=== Processing {dataset} ===")
    #     try:
    #         result = run_discover(dataset)
    #         all_results.append(result)
    #     except Exception as e:
    #         print(f"Error processing {dataset}: {str(e)}")

    # save_combined_results(all_results)
    # print("\nAll experiments completed!")
    sys.stdout = Logger('notes.log', sys.stdout)
    sys.stderr = Logger('notes.log', sys.stderr)
    sga_num = 20
    sga_depth = 4
    sga_width = 5
    sga_p_var = 0.5
    sga_p_mute = 0.3
    sga_p_cro = 0.5
    sga_run = 100

    print('sga_num = ', sga_num)
    print('sga_depth = ', sga_depth)
    print('sga_width = ', sga_width)
    print('sga_p_var = ', sga_p_var)
    print('sga_p_mute = ', sga_p_mute)
    print('sga_p_cro = ', sga_p_cro)
    print('sga_run = ', sga_run)

    sga = SGA(num=sga_num, depth=sga_depth, width=sga_width, p_var=sga_p_var, p_rep=1, p_mute=sga_p_mute, p_cro=sga_p_cro)
    sga.run(sga_run)
