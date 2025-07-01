import pysindy as ps
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data):
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

    elif data == "wave_data.npy" or "kdv_data.npy" in data:
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

u, x, t = load_data("Korteweg-de-Vries/kdv_data.npy")
if len(u.shape) == 2:
    u = u.T.reshape(len(x), len(t), 1)

# functions = [lambda x: np.sin(x), np.cos(x + y)]
# lib_custom = ps.CustomLibrary(library_functions=functions)
pde_lib = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=3,include_bias=False),
                                # function_names=library_function_names,
                                derivative_order=3,
                                spatial_grid=x,
                                # temporal_grid=t,
                                include_bias=False).fit(u)
print(pde_lib.get_feature_names(), "\n")

# optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)
# optimizer = ps.SR3(tol=1e-15, normalize_columns=True, max_iter=10000)
# optimizer = ps.SR3(threshold=5, max_iter=10000, tol=1e-15, thresholder='l1', normalize_columns=True)
# optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
# optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)
optimizer = ps.SSR(criteria='model_residual', normalize_columns=True, kappa=5e-3)

model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
model.fit(u, t=t[1]-t[0])
model.print()