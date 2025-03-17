import numpy as np
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt


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

        return u, u_noisy


class DataLoader:
    def load_data(self, data):
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(directory, "../../data/")
        filepath = os.path.join(directory, data)
        u = np.load(filepath)

        if data == "ode_data.npy" or data == "vdp_data.npy":
            # ode, vdp
            step = 0.05
            steps_num = 320
            t = np.arange(start=0., stop=step * steps_num, step=step)
            x = None
            plt.figure()
            plt.plot(t, u)
            plt.xlabel('t', fontsize=16)
            plt.ylabel('u', fontsize=16)
            plt.title(r'$u(x, t)$', fontsize=16)
            plt.show()

        elif data == "wave_data.npy" or data == "kdv_data.npy":
            # wave, kdv
            shape = len(u)
            t = np.linspace(0, 1, shape)
            x = np.linspace(0, 1, shape)

            plt.figure()
            plt.pcolormesh(x, t, u)
            plt.xlabel('x', fontsize=16)
            plt.ylabel('t', fontsize=16)
            plt.title(r'$u(x, t)$', fontsize=16)
            plt.show()

        elif data == "ac_data.npy":
            # ac
            t = np.linspace(0., 1., 51)
            x = np.linspace(-1., 0.984375, 128)

            plt.figure()
            plt.pcolormesh(x, t, u)
            plt.xlabel('t', fontsize=16)
            plt.ylabel('x', fontsize=16)
            plt.title(r'$u(x, t)$', fontsize=16)
            plt.show()

        return u, x, t

        # step = 0.05
        # steps_num = 320
        # t = np.arange(start=0., stop=step * steps_num, step=step)
        # grids = np.meshgrid(t, t, indexing='ij')  # np.stack(, axis = 2)
        # return data, t, t
