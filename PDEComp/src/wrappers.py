from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import torch
# DeepMoD stuff
from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.constraint import LeastSquares
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic

import odeformer
from odeformer.model import SymbolicTransformerRegressor


# import epde


class PDEDiscoveryWrapper(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def preprocess_data(self, u, x, t):
        """Preprocess the input data."""
        pass

    @abstractmethod
    def discover_pde(self, u, x, t):
        """Discover the PDE from the input data."""
        pass

    @abstractmethod
    def evaluate_performance(self, true_pde, discovered_pde):
        """Evaluate the performance of the discovered PDE."""
        pass

    def run(self, u, x, t, true_pde=None):
        """Run the full PDE discovery pipeline."""
        preprocessed_u, preprocessed_x, preprocessed_t = self.preprocess_data(u, x, t)
        discovered_pde = self.discover_pde(preprocessed_u, preprocessed_x, preprocessed_t)
        if true_pde is not None:
            performance = self.evaluate_performance(true_pde, discovered_pde)
            return discovered_pde, performance
        return discovered_pde


# Example wrapper for PDE-FIND
class PDEFINDWrapper(PDEDiscoveryWrapper):
    def __init__(self):
        super().__init__("PDE-FIND")

    def preprocess_data(self, u, x, t):
        # Implement PDE-FIND specific preprocessing
        if len(u.shape) == 2:
            u = u.T.reshape(len(x), len(t), 1)
        return u, x, t

    def discover_pde(self, u, x, t):
        # Implement PDE-FIND algorithm
        # library_functions = [lambda x: np.cos(x)*np.cos(x)]
        # library_functions = [lambda x: x, lambda x: x * x]
        # library_functions = [lambda x: x * x, lambda x: np.cos(x) * np.cos(x)]
        library_functions = [lambda x: x, lambda x: x * x, lambda x: np.sin(x)*np.cos(x)]

        # library_function_names = [lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']
        # library_function_names = [lambda x: x, lambda x: x + x]
        # library_function_names = [lambda x: x + x, lambda x: 'cos(' + x + ')' + 'sin(' + x + ')']
        library_function_names = [lambda x: x, lambda x: x + x, lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']
        pde_lib = ps.PDELibrary(library_functions=library_functions,
                                function_names=library_function_names,
                                derivative_order=3, spatial_grid=x,
                                include_bias=True,
                                is_uniform=True)

        optimizer = ps.STLSQ(threshold=2, alpha=1e-5, normalize_columns=True)
        # optimizer = ps.SR3(threshold=7, thresholder="l0", tol=1e-15, nu=1e2, normalize_columns=True, max_iter=10000)
        # optimizer = ps.SR3(threshold=5, max_iter=10000, tol=1e-15, thresholder='l1', normalize_columns=True)
        # optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
        # optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)
        # optimizer = ps.SSR(criteria='model_residual', normalize_columns=True, kappa=5e-3)

        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u, t=t[1]-t[0])
        return model.print()
        # print(model.simulate(0, t))
        # return model

    def evaluate_performance(self, true_pde, discovered_pde):
        # Implement performance evaluation
        pass


# Example wrapper for WSINDy
class WSINDyWrapper(PDEDiscoveryWrapper):
    def __init__(self):
        super().__init__("WSINDy")

    def preprocess_data(self, u, x, t):
        # Implement WSINDy specific preprocessing
        if len(u.shape) == 2:
            u = u.T.reshape(len(x), len(t), 1)
        return u, x, t

    def make_grid(self, u, x, t):
        if len(u.shape) == 3:
            X, T = np.meshgrid(x, t)
            XT = np.asarray([X, T]).T
        if len(u.shape) == 1:
            XT = t
        else:
            X, T = np.meshgrid(x, t)
            XT = np.asarray([X, T]).T
        return XT

    def discover_pde(self, u, x, t):
        # Implement WSINDy algorithm
        grid = self.make_grid(u, x, t)

        # library_functions = [lambda x: np.cos(x)*np.cos(x)]
        library_functions = [lambda x: x * x, lambda x: np.cos(x) * np.cos(x)]
        # library_functions = [lambda x: x, lambda x: x * x, lambda x: np.cos(x)*np.cos(x)]
        # library_function_names = [lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']
        library_function_names = [lambda x: x + x, lambda x: 'cos(' + x + ')' + 'sin(' + x + ')']
        # library_function_names = [lambda x: x, lambda x: x + x, lambda x: 'cos(' +x+ ')'+'sin(' +x+ ')']
        pde_lib = ps.WeakPDELibrary(library_functions=library_functions,
                                function_names=library_function_names,
                                derivative_order=3, spatiotemporal_grid=grid,
                                include_bias=True, K=80, is_uniform=True, implicit_terms=False, )

        optimizer = ps.SSR(normalize_columns=True, kappa=1e-10, max_iter=100)
        # optimizer = ps.STLSQ(threshold=5, alpha=1e-3, normalize_columns=True)
        # optimizer = ps.SR3(threshold=0.05, max_iter=1000, tol=1e-3, thresholder='l0', normalize_columns=True)

        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        # model.fit(u)
        model.fit(u, library_ensemble=False, n_models=20)
        model.print()
        return model

    def simulate(self, model, x0, t_test):
        return model.simulate(x0=x0, t=t_test)

    def predict(self, model, x_test):
        return model.predict(x_test)

    def evaluate_performance(self, true_pde, discovered_pde):
        # Implement performance evaluation
        pass


class DeePyMoDWrapper(PDEDiscoveryWrapper):
    def __init__(self):
        super().__init__("DeePyMoD")

    def preprocess_data(self, u, x, t):
        t = np.ravel(t).squeeze()
        x = np.ravel(x).squeeze()
        u = u.T
        return u, x, t

    def discover_pde(self, u, x, t):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        def load_data():
            array = {}
            array["x"], array["t"] = np.meshgrid(x, t, indexing="ij")
            array["u"] = u
            coords = torch.from_numpy(np.stack((array["t"], array["x"]), axis=-1)).float()
            data = torch.from_numpy(np.real(array["u"])).unsqueeze(-1).float()
            return coords, data

        dataset = Dataset(load_data(),
            subsampler=Subsample_random,
            subsampler_kwargs={"number_of_samples": 5000},
            device=device,
        )

        coords = dataset.get_coords().cpu()
        data = dataset.get_data().cpu()
        fig, ax = plt.subplots()
        im = ax.scatter(coords[:, 1], coords[:, 0], c=data[:, 0], marker="x", s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(mappable=im)

        plt.show()

        train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)
        network = NN(2, [30, 30, 30, 30], 1)
        library = Library1D(poly_order=3, diff_order=3)
        estimator = Threshold(0.2)
        sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=200, delta=1e-5)
        constraint = LeastSquares()
        model = DeepMoD(network, library, estimator, constraint).to(device)

        # Defining optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3
        )

        train(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            sparsity_scheduler,
            split=0.8,
            max_iterations=100000,
        )
        print(model.constraint_coeffs())
        print(model.estimator_coeffs())

    def evaluate_performance(self, true_pde, discovered_pde):
        # Implement performance evaluation
        pass


class ODEFormerWrapper(PDEDiscoveryWrapper):
    def __init__(self):
        super().__init__("ODEFormer")

    def preprocess_data(self, u, x, t):
        return u, x, t

    def make_grid(self, u, x, t):
        if len(u.shape) == 3:
            X, T = np.meshgrid(x, t)
            XT = np.asarray([X, T]).T
        if len(u.shape) == 1:
            XT = t
        else:
            X, T = np.meshgrid(x, t)
            XT = np.asarray([X, T]).T
        return XT

    def discover_pde(self, u, x, t):
        dstr = SymbolicTransformerRegressor(
            from_pretrained=True)

        model_args = {'beam_size': 20,
                      'beam_temperature': 0.1}
        dstr.set_model_args(model_args)
        dstr.fit(t, u)
        dstr.print()
        pass

    def evaluate_performance(self, true_pde, discovered_pde):
        # Implement performance evaluation
        pass
