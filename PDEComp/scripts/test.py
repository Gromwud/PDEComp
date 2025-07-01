from PDEComp.src.wrappers import *
from PDEComp.src.data_loader import *


# Usage example
if __name__ == "__main__":
    # Load your data
    dl = DataLoader()
    u, x, t = dl.load_data("data/Korteweg-de-Vries/kdv_data.npy")
    # u, x, t = dl.load_data("wave_data.npy")
    # u, x, t = dl.load_data("ac_data.npy")
    # u, x, t = dl.load_data("vdp_data.npy")
    # u, x, t = dl.load_data("ode_data.npy")

    # u_train = u[:60]
    # u_test = u[60:]
    # x_train = x[:60]
    # x_test = x[60:]
    # t_train = t[:60]
    # t_test = t[60:]

    # Create wrappers
    pde_find = PDEFINDWrapper()
    wsindy = WSINDyWrapper()
    deepymod_model = DeePyMoDWrapper()
    odeformer_model = ODEFormerWrapper()

    # Run PDE discovery
    pde_find_result = pde_find.run(u, x, t)
    # wsindy_result = wsindy.run(u, x, t)
    # deepymod_result = deepymod_model.run(u, x, t)
    # odeformer_result = odeformer_model.run(u, x, t)

    # wsindy_pr = wsindy.predict(wsindy_result, x_test)
    # wsindy.simulate(wsindy_result, np.array([x_test[0]]), t_test)

    # Compare results
    # print("PDE-FIND result:", pde_find_result)
    # print("WSINDy result:", wsindy_result)

