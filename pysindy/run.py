import os
import numpy as np
import pysindy as ps
from pathlib import Path
import scipy.io as scio
import json

# Конфигурация (можно вынести в отдельный config.yaml)
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/pysindy")
DATASETS = ["ac_data.npy", "kdv_data.mat", "burgers_data.mat", "vdp_data.npy"]  # Все тестовые данные

def save_combined_results(results):
    """Save results to a common JSON file"""
    output_file = RESULTS_DIR / "results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    result = []
    
    result.append(results)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

def load_data(filename):
    """Загрузка данных без привязки к структуре базового класса"""
    
    # Универсальная обработка временных данных
    if filename == "ode_data.npy" or filename == "vdp_data.npy":
        data = np.load(DATA_DIR/filename)
        step = 0.05
        steps_num = 320
        t = np.arange(start=0., stop=step * steps_num, step=step)
        x = None

    elif filename == "kdv_data_periodic.npy":
        data = np.load(DATA_DIR/filename)
        shape = len(data)
        t = np.linspace(0, 1, shape)
        x = np.linspace(0, 1, shape)

    elif filename == "ac_data.npy":
        data = np.load(DATA_DIR/filename)
        t = np.linspace(0., 1., 51)
        x = np.linspace(-1., 0.984375, 128)

    elif filename == "kdv_data.mat" or filename == "burgers_data.mat":
        data = scio.loadmat(DATA_DIR/filename)
        t = np.ravel(data['t'])
        x = np.ravel(data['x'])
        data = np.real(data['usol']).T

    if len(data.shape) == 1:
        data = data.T.reshape(len(t), 1)
    elif len(data.shape) == 2:
        data = data.T.reshape(len(x), len(t), 1)

    return data, x, t

def run_sindy(data, x, t, dataset_name):
    """Основная логика идентификации"""
    # 1. Настройка модели
    if dataset_name == "ac_data.npy":
        library = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=3,include_bias=False),
                                # function_names=library_function_names,
                                derivative_order=3,
                                spatial_grid=x,
                                # temporal_grid=t,
                                include_bias=True).fit(data)
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)
        
    elif dataset_name == "kdv_data.mat":
        library = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
                        derivative_order=3, spatial_grid=x,
                        include_bias=True).fit(data)
        
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)

    elif dataset_name == "burgers_data.mat":
        library = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
                        derivative_order=3, spatial_grid=x,
                        include_bias=True).fit(data)
        
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)
    
    elif dataset_name == "vdp_data.npy":
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
                                #normalize_columns=True
                            )
    # 2. Выбор типа задачи
    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    
    # 3. Обучение
    model.fit(data, t=t[1]-t[0])
    model.print(precision=4)
    
    # 4. Сохранение результатов
    result = {
        "dataset": dataset_name.split(".")[0],
        "coefficients": model.coefficients().tolist(),
        "features": model.get_feature_names(),
        # "model_str": str(model.print(precision=4))
    }
    
    # Save individual results
    # save_path = RESULTS_DIR/dataset_name.split(".")[0]
    # save_path.mkdir(parents=True, exist_ok=True)
    # np.save(save_path/"coefficients.npy", model.coefficients())
    
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
    
    # Save combined results
    save_combined_results(all_results)
    print("\nAll experiments completed!")