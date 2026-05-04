epde_params = {
    'burgers_sln_100_data.csv': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': (10, 10),
        'default_preprocessor_type': 'poly',
        'variable_names': ['u', ],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
        'eq_sparsity_interval': (1e-6, 1e-0),
        'fourier_layers': True
    },
    
    'ac_data.npy': {
        'population_size': 8,
        'training_epochs': 30,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-5, 1e-2),
        'fourier_layers': False
    },

    'kdv_data.mat': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-5, 1e-2),
        'fourier_layers': False
    },

    'burgers_data.mat': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False
    },

    'ks_data.mat': {
        'population_size': 16,
        'training_epochs': 10,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 5,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 10,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-0),
        'fourier_layers': False
    },

    'pde_divide_data.npy': {
        'population_size': 8,
        'training_epochs': 50,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'CacheStoredTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-9, 1e-2),
        'fourier_layers': False
    },

    'pde_compound_data.npy': {
        'population_size': 8,
        'training_epochs': 50,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'CacheStoredTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-9, 1e-2),
        'fourier_layers': False
    },

    'lorenz_data.npy': {
        'population_size': 8,
        'training_epochs': 2,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'w'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'TrigonometricTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas' : [0.8, 0.2]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'lotka_data.npy': {
        'population_size': 8,
        'training_epochs': 1,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'TrigonometricTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas' : [0.8, 0.2]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

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
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-4),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'kdv_periodic_data.npy': {
        'population_size': 12,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': 'custom_trig_tokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-10, 1e-2),
        'fourier_layers': False
    },

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
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-8, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'wave_data.csv': {
        'population_size': 8,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'equation_terms_max_number': 5,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-2),
        'fourier_layers': False
    },

    'ns_data.mat': {
        'population_size': 16,
        'training_epochs': 50,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 5,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'p'],
        'equation_terms_max_number': 10,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
        'eq_sparsity_interval': (1e-12, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '3d'
    },

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
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (0.999, 1.001)
    },
}

COMMON_PARAMS = {
    'max_deriv_order': (2, 4),
    'data_fun_pow': 3,
    'equation_factors_max_number': 2
}


sindy_params = {
    'ac_data.npy': {
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 1e-3}
    },
    
    'kdv_data.mat': {
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 5, 'alpha': 1e-5, 'normalize_columns': True}
    },

    'kdv_periodic_data.npy': {
        'crop': 10,
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True, 'custom_tokens': ['sin(x)', 'cos(t)', 'sin(x) cos(t)', 'cos(x) sin(t)']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.05, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 1e-3}
    },

    'burgers_data.mat': {
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 4.5, 'alpha': 1e-3, 'normalize_columns': True}
    },
    
    'burgers_sln_100_data.csv': {
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 1e-5, 'normalize_columns': False, 'coefficient_tol': 0.1}
    },

    'pde_divide_data.npy': {
        'crop': 10,
        'library': {'type': 'pde_custom_concat', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True, 'coordinate_variables': ['x'], 'custom_tokens': ['(1/x) u', '(1/x) u_x']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'pde_compound_data.npy': {
        'crop': 10,
        'library': {'type': 'pde_custom_concat', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True, 'custom_tokens': ['d_x(u u_x)']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.02, 'alpha': 1e-10, 'normalize_columns': False, 'coefficient_tol': 1e-3}
    },

    'ks_data.mat': {
        'crop': 5,
        'library': {'type': 'pde', 'poly_include_bias': False, 'derivative_axes': ['x', 't'], 'pde_include_bias': True, 'diff_kwargs': {'periodic': True}},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 1e-8, 'normalize_columns': True, 'coefficient_tol': 0.03}
    },

    'wave_data.csv': {
        'crop': 10,
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'pde', 'poly_include_bias': True, 'derivative_axes': ['x', 't'], 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'lorenz_data.npy': {
        'library': {'type': 'polynomial', 'data_fun_pow': 2, 'include_bias': True, 'variable_names': ['x0', 'x1', 'x2'], 'polynomial_variables': ['x0', 'x1', 'x2']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 0.05, 'normalize_columns': False}
    },

    'lotka_data.npy': {
        'library': {'type': 'polynomial', 'data_fun_pow': 2, 'include_bias': False, 'variable_names': ['x0', 'x1'], 'polynomial_variables': ['x0', 'x1']},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 0.05, 'normalize_columns': False}
    },

    'vdp_data.npy': {
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'polynomial', 'include_bias': True, 'derivative_axes': ['t']},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 0.02}
    },

    'ode_data.npy': {
        'crop': 10,
        'targets': [{'name': 'u_tt', 'variable': 'u', 'axis': 't', 'order': 2}],
        'library': {'type': 'polynomial', 'include_bias': True, 'derivative_axes': ['t'], 'custom_tokens': ['t', 'u_t sin(2 t)']},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 0.05}
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
            'include_bias': True,
            'max_deriv_order': (1, 2, 2),
            'variable_names': ['u', 'v', 'p'],
        },
        'optimizer': {'type': 'STLSQ', 'threshold': 0.08, 'alpha': 1e-5, 'normalize_columns': True, 'coefficient_tol': 0.01},
        'preprocess': {'moveaxis': True}
    },

    'ODE_simple_discovery': {
        'targets': [{'name': 'u_t', 'variable': 'u', 'axis': 't', 'order': 1}],
        'library': {'type': 'poly_and_fourier', 'poly_include_bias': True, 'derivative_axes': ['t'], 'n_frequencies': 1, 'custom_tokens': ['sin(t)', 'cos(t)']},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    }
}
