import os
import numpy as np
from pathlib import Path
import scipy.io as scio
import json
import torch
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic

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
    
    t = np.ravel(t).squeeze()
    x = np.ravel(x).squeeze()
    data = data.T
    return data, x, t


def run_deepmod(data, x, t, filename):
    
    """Основная логика идентификации"""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    def load_dataset():
        array = {}
        array["x"], array["t"] = np.meshgrid(x, t, indexing="ij")
        array["u"] = data
        coords = torch.from_numpy(np.stack((array["t"], array["x"]), axis=-1)).float()
        u = torch.from_numpy(np.real(array["u"])).unsqueeze(-1).float()
        return coords, u

    dataset = Dataset(load_dataset,
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": 5000},
        device=device,
    )

    if filename == "ac_data.npy":
        train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)
        network = NN(2, [30, 30, 30, 30], 1)
        library = Library1D(poly_order=3, diff_order=3)
        estimator = Threshold(0.1)
        sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
        constraint = LeastSquares()
        model = DeepMoD(network, library, estimator, constraint).to(device)

        # Defining optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
        )

    elif filename == "kdv_data.mat":
        ...

    elif filename == "burgers_data.mat":
        ...

    elif filename == "vdp_data.npy":
        ...

    elif filename == "pde_divide_data.npy":
        ...

    train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        sparsity_scheduler,
        log_dir='deepymod',
        split=0.8,
        max_iterations=100000,
    )

    print(model.constraint_coeffs())
    print(model.estimator_coeffs())

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
            result = run_deepmod(data, x, t, dataset)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
