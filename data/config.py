epde_params = {
    'burgers_sln_100_data.csv': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'ac_data.npy': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'kdv_data.mat': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'burgers_data.mat': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'ks_data.mat': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},
        
    'pde_divide_data.npy': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},
                
    'pde_compound_data.npy': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'lorenz_data.npy': {
        'population_size': 48,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'w'],
        'equation_terms_max_number': 10,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (1.99999999, 2.00000001),
        'multiobjective_mode': True,
        'data_fun_pow': 2,
        'deriv_fun_pow': 1,
        'max_deriv_order': (1,)},

    'lotka_data.npy': {
        'population_size': 8,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v'],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.5, 0.5]},
        'eq_sparsity_interval': (1e-10, 0.1),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (1.99999999, 2.00000001),
        'multiobjective_mode': True,
        'data_fun_pow': 2,
        'deriv_fun_pow': 1,
        'max_deriv_order': (1,)},
                    
    'ode_data.npy': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'TrigonometricTokens, GridTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (1.99999999, 2.00000001),
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2,)},

    'kdv_periodic_data.npy': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens',
                                'TrigonometricTokens',
                                'custom_trig_tokens',
                                'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'vdp_data.npy': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'TrigonometricTokens, GridTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-08, 1.0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (1.99999999, 2.00000001),
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2,)},

    'wave_data.csv': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens', 'sindy_pde_custom_tokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'multiobjective_mode': True,
        'data_fun_pow': 3,
        'deriv_fun_pow': 1,
        'max_deriv_order': (2, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},
                
    'ns_data.mat': {
        'population_size': 48,
        'training_epochs': 5,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 'auto_10pct',
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'p'],
        'equation_terms_max_number': 10,
        'additional_tokens': ['GridTokens', 'TrigonometricTokens'],
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 0.0001),
        'fourier_layers': False,
        'coordinate_tensors': '3d',
        'data_fun_pow': 3,
        'deriv_fun_pow': 2,
        'max_deriv_order': (2, 4, 4),
        'trig_tokens_freq': (1.99999999, 2.00000001)},

    'ODE_simple_discovery': {
        'population_size': 8,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'ODE_simple_discovery',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        'eq_sparsity_interval': (0.0001, 1.0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (0.999, 1.001),
        'max_deriv_order': (2,),
        'data_fun_pow': 3,
        'deriv_fun_pow': 1}}


COMMON_PARAMS = {
    'max_deriv_order': (2, 4),
    'data_fun_pow': 3,
    'equation_factors_max_number': 2,
    'include_bias': True
}


SINDY_PDE_CUSTOM_TOKENS = [
    't',
    'x',
    'sin(x)',
    'cos(t)',
    'sin(x) cos(t)',
    'cos(x) sin(t)',
    '(1/x) u',
    '(1/x) u_x',
    'd_x(u u_x)',
]

SINDY_ODE_CUSTOM_TOKENS = [
    't',
    'sin(t)',
    'cos(t)',
    'u_t sin(2 t)',
]


sindy_params = {
    'ac_data.npy': {
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 3, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },
    
    'kdv_data.mat': {
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 3, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.2}
    },

    'kdv_periodic_data.npy': {
        'crop': 3,
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 1e-5, 'normalize_columns': False, 'coefficient_tol': 0.01}
    },

    'burgers_data.mat': {
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-3, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },
    
    'burgers_sln_100_data.csv': {
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 1, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'pde_divide_data.npy': {
        'crop': 10,
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 8, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.2}
    },

    'pde_compound_data.npy': {
        'crop': 10,
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 3, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.05}
    },

    'ks_data.mat': {
        'crop': 5,
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'diff_kwargs': {'periodic': True}, 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 3, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'wave_data.csv': {
        'crop': 10,
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'pde', 'derivative_axes': ['x', 't'], 'custom_tokens': SINDY_PDE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 3, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'lorenz_data.npy': {
        'library': {'type': 'polynomial', 'data_fun_pow': 2, 'variable_names': ['x0', 'x1', 'x2'], 'polynomial_variables': ['x0', 'x1', 'x2'], 'custom_tokens': ['t']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 0.5, 'normalize_columns': False}
    },

    'lotka_data.npy': {
        'library': {'type': 'polynomial', 'data_fun_pow': 2, 'variable_names': ['x0', 'x1'], 'polynomial_variables': ['x0', 'x1'], 'custom_tokens': ['t']},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 0.5, 'normalize_columns': False}
    },

    'vdp_data.npy': {
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'polynomial', 'derivative_axes': ['t'], 'custom_tokens': SINDY_ODE_CUSTOM_TOKENS},
        'optimizer': {'type': 'FROLS', 'max_iter': 5, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.1}
    },

    'ode_data.npy': {
        'crop': 10,
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'polynomial', 'derivative_axes': ['t'], 'custom_tokens': SINDY_ODE_CUSTOM_TOKENS},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 0.1}
    },

    'ns_data.mat': {
        'targets': [
            {'name': 'u_t', 'variable': 'u', 'axis': 't', 'order': 1},
            {'name': 'v_t', 'variable': 'v', 'axis': 't', 'order': 1},
            {'name': 'u_x', 'variable': 'u', 'axis': 'x', 'order': 1, 'feature_tokens': ['v_y']}
        ],
        'library': {
            'type': 'navier_stokes',
            'data_fun_pow': 1,
            'max_deriv_order': (1, 2, 2),
            'variable_names': ['u', 'v', 'p'],
        },
        'optimizer': {'type': 'STLSQ', 'threshold': 0.08, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'ODE_simple_discovery': {
        'targets': [{'name': 'u_t', 'variable': 'u', 'axis': 't', 'order': 1}],
        'library': {'type': 'poly_and_fourier', 'derivative_axes': ['t'], 'n_frequencies': 1, 'custom_tokens': SINDY_ODE_CUSTOM_TOKENS},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    }
}


DEEPMOD_DEFAULTS = {
    'direct_optimizer': {},
}


deepmod_params = {
    'ode_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-6, 'pdefind_dtol': 1e-4, 'coefficient_tol': 0.5},
    },
    'vdp_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-12, 'pdefind_dtol': 1e-3, 'coefficient_tol': 0.0},
    },
    'lorenz_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-12, 'pdefind_dtol': 0.1, 'coefficient_tol': 0.0},
    },
    'lotka_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-6, 'pdefind_dtol': 1e-3, 'coefficient_tol': 0.0},
    },
    'burgers_data.mat': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-12, 'pdefind_dtol': 0.01, 'coefficient_tol': 0.0},
    },
    'ac_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 0.01, 'pdefind_dtol': 1e-3, 'coefficient_tol': 0.01},
    },
    'kdv_data.mat': {
        'direct_optimizer': {'type': 'threshold', 'threshold': 1e-6, 'coefficient_tol': 0.5},
    },
    'kdv_periodic_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-12, 'pdefind_dtol': 1e-3, 'coefficient_tol': 0.0},
    },
    'wave_data.csv': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 0.01, 'pdefind_dtol': 0.5, 'coefficient_tol': 0.0},
    },
    'pde_divide_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-6, 'pdefind_dtol': 0.5, 'coefficient_tol': 0.0},
    },
    'pde_compound_data.npy': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-4, 'pdefind_dtol': 0.5, 'coefficient_tol': 0.0},
    },
    'ns_data.mat': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 0.01, 'pdefind_dtol': 0.5, 'coefficient_tol': 0.0},
    },
    'ks_data.mat': {
        'direct_optimizer': {'type': 'threshold', 'threshold': 0.05, 'coefficient_tol': 0.05},
    },
    'burgers_sln_100_data.csv': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-4, 'pdefind_dtol': 0.5, 'coefficient_tol': 0.0},
    },
    'ODE_simple_discovery': {
        'direct_optimizer': {'type': 'pdefind', 'pdefind_lam': 1e-8, 'pdefind_dtol': 1e-3, 'coefficient_tol': 0.0},
    },
}


DEEPMOD_DATASETS = list(deepmod_params.keys())


EDL_DEFAULTS = {
    'optimizer': {
        'type': 'STRidge',
        'lam': 1e-5,
        'tol': 0.1,
        'str_iters': 10,
        'normalize': 2,
        'coefficient_tol': 0.0,
    },
}


edl_params = {
    'ode_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 5.0, 'normalize': 2, 'coefficient_tol': 0.01},
    },
    'vdp_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1e-4, 'tol': 1.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'lorenz_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 0.05, 'normalize': 0, 'coefficient_tol': 0.01},
    },
    'lotka_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1e-4, 'tol': 100.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'burgers_data.mat': {
        'optimizer': {'type': 'STRidge', 'lam': 0.01, 'tol': 2.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'ac_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 2.0, 'normalize': 2, 'coefficient_tol': 0.01},
    },
    'kdv_data.mat': {
        'optimizer': {'type': 'STRidge', 'lam': 0.01, 'tol': 5.0, 'normalize': 2, 'coefficient_tol': 0.02},
    },
    'kdv_periodic_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1e-4, 'tol': 1.0, 'normalize': 2, 'coefficient_tol': 0.001},
    },
    'wave_data.csv': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 20.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'pde_divide_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1e-4, 'tol': 5.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'pde_compound_data.npy': {
        'optimizer': {'type': 'STRidge', 'lam': 1e-4, 'tol': 10.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'ns_data.mat': {
        'optimizer': {'type': 'STRidge', 'lam': 0.01, 'tol': 1.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
    'ks_data.mat': {
        'optimizer': {'type': 'STRidge', 'lam': 0.01, 'tol': 10.0, 'normalize': 2, 'coefficient_tol': 0.005},
    },
    'burgers_sln_100_data.csv': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 1e-6, 'normalize': 0, 'coefficient_tol': 0.02},
    },
    'ODE_simple_discovery': {
        'optimizer': {'type': 'STRidge', 'lam': 1.0, 'tol': 1.0, 'normalize': 2, 'coefficient_tol': 0.0},
    },
}


EDL_DATASETS = list(edl_params.keys())


discover_params = {
    'ode_data.npy': {
        'base_config': 'config_pde_Burgers.json',
        'function_set': ['add'],
        'target': 'u_tt',
    },
    'vdp_data.npy': {
        'base_config': 'config_pde_Burgers.json',
        'function_set': ['add'],
        'target': 'u_tt',
    },
    'ODE_simple_discovery': {
        'base_config': 'config_pde_Burgers.json',
        'function_set': ['add'],
        'target': 'u_t',
    },
    'burgers_data.mat': {
        'base_config': 'config_pde_Burgers.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_t',
    },
    'burgers_sln_100_data.csv': {
        'base_config': 'config_pde_Burgers.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_t',
    },
    'ac_data.npy': {
        'base_config': 'config_pde_Chafee.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_t',
    },
    'kdv_data.mat': {
        'base_config': 'config_pde_KdV.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'diff3', 'n2', 'n3'],
        'target': 'u_t',
    },
    'kdv_periodic_data.npy': {
        'base_config': 'config_pde_KdV.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'diff3', 'sin', 'cos', 'n2', 'n3'],
        'target': 'u_t',
        'n_samples': 10000,
        'batch_size': 250,
        'max_length': 5,
        'max_add_count': 2,
    },
    'wave_data.csv': {
        'base_config': 'config_pde_wave.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_tt',
    },
    'pde_divide_data.npy': {
        'base_config': 'config_pde_Divide.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_t',
    },
    'pde_compound_data.npy': {
        'base_config': 'config_pde_Compound.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'n2', 'n3'],
        'target': 'u_t',
    },
    'ks_data.mat': {
        'base_config': 'config_pde_KdV.json',
        'function_set': ['add', 'mul', 'div', 'diff', 'diff2', 'diff3', 'diff4', 'n2', 'n3'],
        'target': 'u_t',
    },
}


DISCOVER_DEFAULTS = {
    'n_samples': 5000,
    'batch_size': 250,
    'epsilon': 0.05,
    'n_cores_batch': 1,
    'early_stopping': False,
    'verbose': False,
}


DISCOVER_DATASETS = list(discover_params.keys())

TRUE_COEFFICIENTS = {
    "ode_data.npy": {
        "u_tt": {
            "u_t sin(2 t)": -1.0,
            "t": 1.5,
            "u": -4.0,
        },
    },
    "vdp_data.npy": {
        "u_tt": {
            "u_t": 0.2,
            "u^2 u_t": -0.2,
            "u": -1.0,
        },
    },
    "lotka_data.npy": {
        "x0_t": {
            "x0": 20.0,
            "x0 x1": -20.0,
        },
        "x1_t": {
            "x0 x1": 20.0,
            "x1": -20.0,
        },
    },
    "lorenz_data.npy": {
        "x0_t": {
            "x0": -10.0,
            "x1": 10.0,
        },
        "x1_t": {
            "x0": 28.0,
            "x0 x2": -1.0,
            "x1": -1.0,
        },
        "x2_t": {
            "x0 x1": 1.0,
            "x2": -8.0 / 3.0,
        },
    },
    "burgers_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": 0.1,
        },
    },
    "burgers_sln_100_data.csv": {
        "u_t": {
            "u u_x": -1.0,
        },
    },
    "ac_data.npy": {
        "u_t": {
            "u^3": -5.0,
            "u": 5.0,
        },
    },
    "kdv_data.mat": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
        },
    },
    "kdv_periodic_data.npy": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
            "sin(x) cos(t)": 1.0,
        },
    },
    "wave_data.csv": {
        "u_tt": {
            "u_xx": 0.04,
        },
    },
    "pde_compound_data.npy": {
        "u_t": {
            "d_x(u u_x)": 1.0,
        },
    },
    "pde_divide_data.npy": {
        "u_t": {
            "u_xx": 0.25,
            "(1/x) u_x": -1.0,
        },
    },
    "ks_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": -1.0,
            "u_xxxx": -1.0,
        },
    },
    "ns_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "v u_y": -1.0,
            "p_x": -1.0,
            "(u_xx + u_yy)": 0.01,
        },
        "v_t": {
            "u v_x": -1.0,
            "v v_y": -1.0,
            "p_y": -1.0,
            "(v_xx + v_yy)": 0.01,
        },
        "u_x": {
            "v_y": -1.0,
        },
    },
    "ODE_simple_discovery": {
        "u_t": {
            "cos(t)": 1.0,
            "sin(t)": -1.3,
        },
    },
}

TRUE_COEFFICIENT_ALTERNATIVES = {
    "ODE_simple_discovery": {
        "u_t": [
            {
                "u": 1.0 / 1.3,
                "sin(t)": -(1.0 / 1.3 + 1.3),
            },
        ],
    },
}
