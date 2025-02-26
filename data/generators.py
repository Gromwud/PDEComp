import numpy as np
from scipy.integrate import solve_ivp
import os


class PDEDataGenerator:
    def generate_burgers(self, x, t, nu=0.1, noise_level=0.05):
        # Generate synthetic Burgers' equation data
        def burgers_pde(t, u):
            ux = np.gradient(u, x, axis=0)
            uxx = np.gradient(ux, x, axis=0)
            return -u * ux + nu * uxx

        u0 = np.exp(-(x**2) / 0.1)  # Initial condition
        sol = solve_ivp(burgers_pde, [t.min(), t.max()], u0, t_eval=t)
        u = sol.y.T
        u_noisy = u + noise_level * np.std(u) * np.random.randn(*u.shape)

        true_coeffs = {
            "u": 0,  # Coefficient for u (if present)
            "u_x": 1,  # Coefficient for u*u_x
            "u_xx": -nu,  # Coefficient for diffusion term
        }

        return u, u_noisy, true_coeffs


class DataInputer:
    def select_data(self, data):
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(directory, "datasets")
        self.filepath = os.path.join(directory, data)

    def extract_data(self):
        data = np.loadtxt(self.filepath, delimiter=',').T
        shape = len(data)
        t = np.linspace(0, 1, shape)
        x = np.linspace(0, 1, shape)
        grids = np.meshgrid(x, t, indexing='ij')  # np.stack(, axis = 2)

        # data = np.load(self.filepath)
        # step = 0.05
        # steps_num = 320
        # t = np.arange(start=0., stop=step * steps_num, step=step)
        # grids = np.meshgrid(t, t, indexing='ij')  # np.stack(, axis = 2)
        return grids, data
