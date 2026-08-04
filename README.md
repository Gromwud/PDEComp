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
sources are mounted as git submodules in `deepmod/deepymod/` and
`discover/discover/`. DISCOVER currently supports the scalar 1D PDE datasets
configured in `data/config.py`. Its data loader is patched to receive benchmark
data and derivatives from the shared `utils/` layer.

## Docker

The main benchmark image runs PySINDy, DeepMoD and EPDE:

```powershell
docker compose build
docker compose run --rm benchmark
```

The default command runs clean metrics for all frameworks. Any benchmark script
can be run through the same container:

```powershell
docker compose run --rm benchmark python clean_run_metrics.py pysindy
docker compose run --rm benchmark python noise_test.py deepmod --datasets ac_data.npy --levels 10 15 20
docker compose run --rm benchmark python noise_boundary_metrics.py pysindy --boundaries-csv results/pysindy_noisy/noise_manual_3_5_summary.csv
```

DISCOVER uses a separate image because it depends on the old TensorFlow 1.x
stack:

```powershell
docker compose build discover
docker compose run --rm discover
docker compose run --rm discover python noise_test.py discover --datasets burgers_data.mat --levels 0 1 2
```

Both containers mount `data/` as read-only and write outputs to `results/`.

## Metrics

Clean runs report:

- `runtime_seconds`: wall-clock time for one clean run;
- `library_size`: number of candidate terms;
- `relative_error_sum`: sum of relative coefficient errors for expected terms;
- `missing_terms` and `extra_terms`: structural differences against ground truth.

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

The current clean and noisy summaries are collected in `results.md`.
