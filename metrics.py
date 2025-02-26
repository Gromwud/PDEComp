import numpy as np


def coefficient_error(true_coeffs, pred_coeffs):
    return np.mean(np.abs(true_coeffs - pred_coeffs))


def prediction_rmse(true_data, pred_data):
    return np.sqrt(np.mean((true_data - pred_data)**2))
