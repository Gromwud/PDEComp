from metrics import coefficient_error, prediction_rmse
import numpy as np
import matplotlib.pyplot as plt


def run_experiment(data, model, noise_level):

    # Train model
    model.fit()
    model.print()

    # Plot the fit of the derivative
    # x_dot_train = model.differentiate()
    # x_dot_train_pred = model.predict() # will default to d/dt of all the features
    # plt.figure()
    # step = 0.05
    # steps_num = 320
    # t = np.arange(start=0., stop=step * steps_num, step=step)
    # plt.plot(t, x_dot_train, 'k')
    # plt.plot(t, x_dot_train_pred, 'r--')

    # u_pred = model.simulate()
    # return u_pred

    # Evaluate
    # error = coefficient_error(list(true_coeffs.values()), coeffs)
    # return {"error": error, "coeffs": coeffs}

