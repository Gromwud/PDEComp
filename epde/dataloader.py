import numpy as np
from pathlib import Path
import scipy.io as scio
import pandas as pd
DATA_DIR = Path("data")


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

        t = t[:t_train]

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