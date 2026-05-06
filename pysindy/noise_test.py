from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path().absolute()))

from data.dataloader import load_data
import run as sindy_run


NOISY_RESULTS_DIR = Path("results/pysindy_noisy")

NOISE_LEVELS = {
    "ode_data.npy": 1,
    "vdp_data.npy": 1,
    "lorenz_data.npy": 1,
    "lotka_data.npy": 1,
    "burgers_data.mat": 1,
    "ac_data.npy": 1,
    "kdv_data.mat": 1,
    "kdv_periodic_data.npy": 1,
    "wave_data.csv": 1,
    "pde_divide_data.npy": 1,
    "pde_compound_data.npy": 1,
    "ns_data.mat": 1,
    "ks_data.mat": 1,
    "burgers_sln_100_data.csv": 1,
    "ODE_simple_discovery": 1,
}

DEFAULT_NOISE_SCALE = 0.01


def add_noise(data, noise_level, scale=DEFAULT_NOISE_SCALE):
    """Add Gaussian noise proportional to data std."""

    return noise_level * scale * np.std(data) * np.random.normal(size=data.shape) + data


def add_dataset_noise(data, filename, noise_level):
    """Apply noise independently to every data variable."""

    if isinstance(data, list):
        return [
            add_noise(values, noise_level)
            for values in data
        ]
    return add_noise(data, noise_level)


def run_noisy_sindy(dataset):
    """Load one dataset, add configured noise, and run PySINDy."""

    noise_level = NOISE_LEVELS.get(dataset, 0)
    data, x, y, z, t = load_data(dataset)
    noised_data = add_dataset_noise(data, dataset, noise_level)

    result = sindy_run.run_sindy(noised_data, x, y, z, t, dataset)
    result["noise_level"] = noise_level
    result["noise_scale"] = DEFAULT_NOISE_SCALE
    return result


if __name__ == "__main__":
    sindy_run.RESULTS_DIR = NOISY_RESULTS_DIR

    all_results = []
    for dataset in sindy_run.DATASETS:
        print(f"\n=== Processing noisy pysindy {dataset} ===")
        try:
            all_results.append(run_noisy_sindy(dataset))
        except Exception as error:
            print(f"Error processing noisy {dataset}: {error}")

    sindy_run.save_combined_results(all_results)
    print("\nAll noisy PySINDy experiments completed!")
