from data.generators import PDEDataGenerator, DataInputer
from models.sindy import SINDyModel, PDEFIND
from experiments.experiments import run_experiment
import matplotlib.pyplot as plt


generator = PDEDataGenerator()
inputer = DataInputer()

inputer.select_data('KdV_data.csv')
# inputer.select_data('vdp_data.npy')


grids, data = inputer.extract_data()
x = grids[1][0]
t = grids[0][:, 0]

models = {
    # "SINDy": SINDyModel(alpha=0.1),
    "PDE-FIND": PDEFIND(data, x, t),
    # Add other models (e.g., PINNs, DeepMoD)
}

results = {}
for model_name, model in models.items():
    results[model_name] = run_experiment(
        data=data,
        model=model,
        noise_level=0
    )

# Visualize results
plt.figure()
plt.pcolormesh(t, x, data)
# plt.plot(data, t)
plt.xlabel('t', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title(r'$u(x, t)$', fontsize=16)
plt.show()

