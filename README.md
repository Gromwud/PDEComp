# DEComp Benchmark

DEComp is a benchmark for comparing equation discovery frameworks on the
same synthetic ODE/PDE datasets. The current working comparison focuses on
EPDE, PySINDy, DeepMoD and DISCOVER.

## What Is Compared

The benchmark uses the datasets and true coefficients from
`data/config.py`. Each framework is evaluated on the same loaded data from
`utils/dataloader.py`.

For a fair comparison, frameworks are configured to use:

- the same polynomial terms;
- the same derivative orders;
- the same custom tokens for 1D and 2D PDEs;
- the same special library shape for systems such as Navier-Stokes.

## Framework Integration

The benchmark keeps data loading, derivative calculation, library construction,
and metric calculation outside individual framework wrappers. This keeps the
comparison focused on how each framework selects equation terms and estimates
coefficients.

- **PySINDy** is used as the direct sparse-regression baseline. The wrapper
  builds the shared candidate library from `utils/sindy_library.py`, computes
  derivatives with `utils/derivatives.py`, and applies the optimizer configured
  in `data/config.py`.
- **DeepMoD** is run through its sparse estimators on the same precomputed
  NumPy feature matrices and target derivatives. The wrapper uses DeePyMoD's
  optimizer classes, then maps selected terms back to the shared benchmark
  feature names for metrics.
- **EPDE** is run through its evolutionary search pipeline. Dataset-specific
  EPDE settings are stored directly in `data/config.py`; the wrapper prepares
  benchmark data, coordinate tensors, derivatives, and custom token families so
  EPDE searches in the comparable term space.
- **DISCOVER** is connected as a separate submodule and container because it
  uses a TensorFlow 1.x stack. The benchmark uses its external fixed-library
  mode: shared features are passed as fixed `theta_*` tokens, and DISCOVER
  searches symbolic combinations of those tokens. In the current wrapper it is
  used for scalar ODE and scalar 1D PDE datasets, not for systems.

## Main Scripts

`clean_run_metrics.py` measures clean-data runs for one framework:

```powershell
python clean_run_metrics.py pysindy
```

The output CSV contains runtime, library size, discovered active terms, expected
terms, and relative coefficient error.

`noise_test.py` is the runner with noisy data:

```powershell
python noise_test.py pysindy --datasets ode_data.npy --levels 0.5 0.75 1.0
python noise_test.py deepmod --datasets ac_data.npy --levels 10 15 20
```

Noise is Gaussian and proportional to the data standard deviation:

```text
u_noisy = u + noise_level * 0.01 * std(u) * np.random.normal()
```

For systems, the summary row `__system__` is counted as correct only when all
component equations have the same structure as the clean run.

`noise_boundary_metrics.py` measures structural and coefficient errors at fixed
noise boundaries:

```powershell
python noise_boundary_metrics.py pysindy --boundaries-csv results\pysindy_noisy\noise_manual_3_5_summary.csv
python noise_boundary_metrics.py deepmod --boundaries-csv results\deepmod\noise_manual_3_5_summary_noise_tuned.csv
```

It reports HD across all noisy runs and RE only for structurally correct runs.

`deepmod/run.py` and `discover/run.py` are benchmark wrappers. Their framework
sources are checked out as git submodules in `deepmod/deepymod/` and
`discover/discover/`. DISCOVER supports the configured scalar ODE and 1D PDE
datasets through an external fixed-library mode. In this mode the shared
`utils/` layer loads the data, computes the target derivative, and builds the
same candidate library used by the other frameworks. DISCOVER then searches over
fixed `theta_*` library terms instead of relying on its built-in PDE derivative
tokens. For ODE datasets, the wrapper passes a dummy spatial coordinate only to
fit DISCOVER's PDE-task interface; the actual ODE library and target are still
computed from the benchmark data.

## Docker

Each framework has its own Docker image and Python environment:

```text
pysindy   -> PySINDy dependencies
deepmod   -> DeePyMoD dependencies
epde      -> EPDE dependencies
discover  -> DISCOVER with TensorFlow 1.x
```

Build all images:

```powershell
docker compose build
```

Run the default clean metrics for one framework:

```powershell
docker compose run --rm pysindy
docker compose run --rm deepmod
docker compose run --rm epde
docker compose run --rm discover
```

Any benchmark script can be run in the matching framework container:

```powershell
docker compose run --rm pysindy python noise_test.py pysindy --datasets ode_data.npy --levels 0.5 1.0
docker compose run --rm deepmod python noise_test.py deepmod --datasets ac_data.npy --levels 10 15 20
docker compose run --rm epde python clean_run_metrics.py epde --datasets wave_data.csv
docker compose run --rm discover python noise_boundary_metrics.py discover --boundaries-csv results/discover/noise_boundaries_3_5.csv
```

All framework containers mount `data/` as read-only and write outputs to the
shared `results/` directory. DISCOVER stays separate because it depends on the
old TensorFlow 1.x stack, which conflicts with the modern environments used by
the other frameworks.

## Metrics

Clean runs report:

- `runtime_seconds`: wall-clock time for one clean run;
- `library_size`: number of candidate terms;
- `relative_error_sum`: sum of relative coefficient errors for expected terms;
- `missing_terms` and `extra_terms`: structural differences against ground truth.

Some datasets can define equivalent coefficient forms in
`TRUE_COEFFICIENT_ALTERNATIVES`. Metrics choose the structurally closest
accepted form, then the one with the smallest coefficient error. This accepted
form logic is used by both `clean_run_metrics.py` and
`noise_boundary_metrics.py`; `noise_test.py` still searches for repeatability of
the clean-run structure.

Noisy runs report:

- `success_count`: how many of `runs` recovered the clean-run structure;

Boundary noise metrics from `noise_boundary_metrics.py` report:

- `correct_count`: how many noisy runs recovered the expected structure;
- `hd_mean` and `hd_std`: mean and standard deviation of the Hamming distance
  over all completed noisy runs;
- `re_mean` and `re_std`: mean and standard deviation of the relative
  coefficient error, computed only for structurally correct runs;

Here HD counts structural mistakes, for example missing or extra active terms.
RE is separated from HD because coefficient comparison is meaningful only when
the discovered equation has the correct structure.


## Current Results

The current clean and noisy summaries are collected in [`results.md`](results.md).
