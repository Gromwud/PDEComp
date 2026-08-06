# Current Benchmark Results

Current PySINDy, DeepMoD, and DISCOVER comparison. For multi-equation systems,
only the aggregate `system` row is shown.

## Clean Runs

### PySINDy

| Dataset | Target | Time, s | Library | RE sum |
|---|---:|---:|---:|---:|
| ode_data.npy | u_tt | 0.00136 | 12 | 0.0164969 |
| vdp_data.npy | u_tt | 0.00315 | 12 | 0.11788 |
| lorenz_data.npy | system | 0.006855 | 60 | 0.0051757 |
| lotka_data.npy | system | 0.002123 | 20 | 0.0232233 |
| burgers_data.mat | u_t | 0.04477 | 33 | 0.0180568 |
| ac_data.npy | u_t | 0.02161 | 33 | 0.00616941 |
| kdv_data.mat | u_t | 0.3588 | 33 | 0.0160575 |
| kdv_periodic_data.npy | u_t | 0.01181 | 33 | 0.000582275 |
| wave_data.csv | u_tt | 0.01568 | 33 | 0.016995 |
| pde_divide_data.npy | u_t | 0.0797 | 33 | 0.00208748 |
| pde_compound_data.npy | u_t | 0.0547 | 33 | 0.00139669 |
| ns_data.mat | system | 0.5719 | 23 | 0.534243 |
| ks_data.mat | u_t | 0.9957 | 33 | 0.0359753 |
| burgers_sln_100_data.csv | u_t | 0.02009 | 33 | 0.000269232 |
| ODE_simple_discovery | u_t | 0.001256 | 11 | 0.00134893 |

### DeepMoD

DeepMoD uses the shared NumPy derivative pipeline and the same candidate
libraries as PySINDy. The sparse step uses DeepMoD's `PDEFIND`/`Threshold`
optimizers.

| Dataset | Target | Time, s | Library | RE sum |
|---|---:|---:|---:|---:|
| ode_data.npy | u_tt | 0.0769 | 12 | 0.0156623 |
| vdp_data.npy | u_tt | 0.0799 | 12 | 0.0683646 |
| lorenz_data.npy | system | 0.434 | 60 | 0.000766131 |
| lotka_data.npy | system | 0.1748 | 20 | 0.0141232 |
| burgers_data.mat | u_t | 2.582 | 33 | 0.0185416 |
| ac_data.npy | u_t | 0.8297 | 33 | 0.00184694 |
| kdv_data.mat | u_t | 4.841 | 33 | 0.00241649 |
| kdv_periodic_data.npy | u_t | 0.7829 | 33 | 0.00640046 |
| wave_data.csv | u_tt | 0.4665 | 33 | 0.00300939 |
| pde_divide_data.npy | u_t | 1.562 | 33 | 0.000366079 |
| pde_compound_data.npy | u_t | 1.53 | 33 | 0.000299912 |
| ns_data.mat | system | 12.25 | 23 | 0.460828 |
| ks_data.mat | u_t | 8.742 | 33 | 0.0221569 |
| burgers_sln_100_data.csv | u_t | 0.9585 | 33 | 0.000259235 |
| ODE_simple_discovery | u_t | 0.0844 | 11 | 0.00132894 |

### DISCOVER

DISCOVER is currently integrated for scalar ODE and scalar 1D PDE datasets. It
uses the same fixed candidate libraries as PySINDy/DeepMoD.

| Dataset | Target | Correct structure | Time, s | Library | RE sum |
|---|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | yes | 11.31 | 12 | 0.0166199 |
| vdp_data.npy | u_tt | yes | 5.38 | 12 | 0.0683646 |
| burgers_data.mat | u_t | yes | 25.50 | 33 | 0.0185416 |
| burgers_sln_100_data.csv | u_t | yes | 16.65 | 33 | 0.00395993 |
| ac_data.npy | u_t | yes | 17.01 | 33 | 0.0133155 |
| kdv_data.mat | u_t | yes | 37.38 | 33 | 0.0303891 |
| kdv_periodic_data.npy | u_t | yes | 23.78 | 33 | 0.00640046 |
| wave_data.csv | u_tt | yes | 10.22 | 33 | 0.0103913 |
| pde_divide_data.npy | u_t | yes | 23.19 | 33 | 0.000366079 |
| pde_compound_data.npy | u_t | yes | 18.95 | 33 | 0.000535998 |
| ks_data.mat | u_t | yes | 66.12 | 33 | 0.036014 |
| ODE_simple_discovery | u_t | yes | 11.95 | 11 | 0.00132894 |

For `ODE_simple_discovery`, DISCOVER finds `sin(t)` and `u` on clean data
instead of the benchmark tokens `sin(t)` and `cos(t)`. Since the data itself is
`u = sin(t) + 1.3 cos(t)`, this is accepted as an equivalent coefficient form.

## Noise Boundaries

The table shows noise levels where 3-5 runs out of 30 recover the clean-run
structure. Larger values mean the method remained structurally stable at higher
noise for that dataset.

`HD` is averaged across all 30 noisy runs. `RE` is averaged only across
structurally correct runs, where `HD = 0`.

### PySINDy

| Dataset | Target | Noise level | Correct | HD mean | HD std | RE mean | RE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | 0.88 | 3/30 | 1.733 | 1.048 | 0.08525 | 0.02673 |
| vdp_data.npy | u_tt | 0.486 | 4/30 | 2.633 | 1.402 | 0.105 | 0.04196 |
| lorenz_data.npy | system | 0.72 | 3/30 | 5.533 | 3.73 | 0.0281 | 0.0179 |
| lotka_data.npy | system | 4 | 4/30 | 4.867 | 3.803 | 0.06733 | 0.009371 |
| burgers_data.mat | u_t | 1.1 | 3/30 | 4.533 | 1.634 | 0.08212 | 0.00164 |
| ac_data.npy | u_t | 21.5 | 3/30 | 1.233 | 0.6261 | 1.236 | 0.05365 |
| kdv_data.mat | u_t | 0.0124 | 3/30 | 0.9 | 0.3051 | 0.469 | 0.0002508 |
| kdv_periodic_data.npy | u_t | 2.7e-05 | 5/30 | 9.1 | 4.566 | 0.003963 | 0.0002696 |
| wave_data.csv | u_tt | 0.498 | 3/30 | 0.9 | 0.3051 | 0.7385 | 0.03028 |
| pde_divide_data.npy | u_t | 0.0084 | 3/30 | 4.5 | 1.526 | 0.0845 | 0.0004646 |
| pde_compound_data.npy | u_t | 0.0452 | 5/30 | 0.8333 | 0.379 | 0.009038 | 5.796e-05 |
| ns_data.mat | system | 0.485 | 4/30 | 0.8667 | 0.3457 | 0.2567 | 0.002665 |
| ks_data.mat | u_t | 0.007155 | 5/30 | 1.667 | 0.7581 | 2.31 | 0.000682 |
| burgers_sln_100_data.csv | u_t | 0.97 | 3/30 | 1.8 | 0.6103 | 0.03663 | 0.001138 |
| ODE_simple_discovery | u_t | 10 | 3/30 | 3.633 | 2.189 | 0.04757 | 0.0178 |

### DeepMoD

DeepMoD sparse optimizer configs were tuned for noise robustness. The candidate
libraries are unchanged.

| Dataset | Target | Noise level | Correct | HD mean | HD std | RE mean | RE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | 2.2 | 4/30 | 1.033 | 0.7184 | 0.41 | 0.04924 |
| vdp_data.npy | u_tt | 0.22 | 3/30 | 5.767 | 3.277 | 0.0909 | 0.04182 |
| lorenz_data.npy | system | 0.0002 | 5/30 | 1.2 | 0.7144 | 0.000765 | 6.958e-06 |
| lotka_data.npy | system | 1.5 | 3/30 | 7.533 | 4.911 | 0.02674 | 0.006807 |
| burgers_data.mat | u_t | 0.092 | 5/30 | 4.7 | 5.2 | 0.009666 | 0.0002555 |
| ac_data.npy | u_t | 21.3 | 3/30 | 5.867 | 5.029 | 1.258 | 0.02157 |
| kdv_data.mat | u_t | 0.0062 | 5/30 | 1.667 | 0.7581 | 0.1484 | 0.000346 |
| kdv_periodic_data.npy | u_t | 9.5e-06 | 3/30 | 20.17 | 9.91 | 0.006078 | 0.000215 |
| wave_data.csv | u_tt | 0.026 | 4/30 | 22.7 | 9.252 | 0.02151 | 0.003029 |
| pde_divide_data.npy | u_t | 0.0101 | 5/30 | 4.167 | 1.895 | 0.001658 | 4.102e-05 |
| pde_compound_data.npy | u_t | 0.127 | 3/30 | 1.8 | 0.6103 | 0.1977 | 0.003249 |
| ns_data.mat | system | 0.79 | 3/30 | 1.233 | 0.6261 | 0.1898 | 0.0027 |
| ks_data.mat | u_t | 0.00177 | 3/30 | 0.9 | 0.3051 | 0.5524 | 0.01036 |
| burgers_sln_100_data.csv | u_t | 0.45 | 3/30 | 1.8 | 0.6103 | 0.009085 | 0.000408 |
| ODE_simple_discovery | u_t | 35 | 3/30 | 4.033 | 2.484 | 0.1014 | 0.06268 |

### DISCOVER

DISCOVER noise metrics were measured for scalar ODE and scalar 1D PDE datasets
supported by the current wrapper.

| Dataset | Target | Noise level | Correct | HD mean | HD std | RE mean | RE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | 1.1 | 3/30 | 0.9 | 0.3051 | 0.05036 | 0.004817 |
| vdp_data.npy | u_tt | 0.14 | 4/30 | 2.6 | 1.037 | 0.05988 | 0.03242 |
| burgers_data.mat | u_t | 0.515 | 5/30 | 1.133 | 0.6814 | 0.2459 | 0.002784 |
| burgers_sln_100_data.csv | u_t | 0.00365 | 5/30 | 0.8333 | 0.379 | 0.003217 | 6.648e-05 |
| ac_data.npy | u_t | 20.5 | 5/30 | 1.067 | 0.6397 | 1.301 | 0.04195 |
| kdv_data.mat | u_t | 0.024 | 3/30 | 1.8 | 0.6103 | 0.2816 | 0.0003275 |
| kdv_periodic_data.npy | u_t | 2.2e-05 | 3/30 | 1.8 | 0.6103 | 0.00427 | 2.838e-05 |
| wave_data.csv | u_tt | 0.0255 | 3/30 | 0.9 | 0.3051 | 0.09405 | 0.01078 |
| pde_divide_data.npy | u_t | 0.0262 | 5/30 | 3.333 | 1.516 | 0.01283 | 0.0001013 |
| pde_compound_data.npy | u_t | 0.1795 | 4/30 | 2.6 | 1.037 | 0.02988 | 0.0001233 |
| ks_data.mat | u_t | 0.01027 | 4/30 | 0.8667 | 0.3457 | 2.599 | 0.0003842 |
| ODE_simple_discovery | u_t | 110 | 3/30 | 2.533 | 1.279 | 0.3281 | 0.1505 |

For `ODE_simple_discovery`, both `cos(t), sin(t)` and the equivalent
`u, sin(t)` form are counted as correct.

The same equivalent-form rule was also checked for PySINDy and DeepMoD.
PySINDy still reaches the 3/30 boundary at noise 10, while DeepMoD's accepted
boundary increases to noise 35 because several noisy runs use the equivalent
`u, sin(t)` representation.

## Conclusions

- On clean data, all three compared frameworks recover the configured scalar
  equations that they currently support. PySINDy is the fastest baseline,
  DeepMoD is slower because it uses its own sparse-estimator path, and DISCOVER
  is the slowest because it performs symbolic search with a TensorFlow 1.x
  reinforcement-learning loop.
- PySINDy gives stable and very fast clean runs across the full benchmark,
  including ODEs, PDEs, and systems. Its main weak points under noise are
  high-order derivative datasets such as `kdv_periodic_data.npy`, `ks_data.mat`,
  and some coefficient errors at the noise boundary.
- DeepMoD matches or improves clean coefficient errors on several datasets, but
  its noise robustness is uneven. It is strong on `ode_data.npy`, `ns_data.mat`,
  and `pde_compound_data.npy`, but much weaker than PySINDy on
  `burgers_data.mat`, `wave_data.csv`, and the periodic KdV case.
- DISCOVER works well in the current benchmark for scalar ODE and scalar 1D PDE
  datasets through the fixed-library wrapper. It is competitive on several PDE
  noise thresholds, especially `kdv_data.mat`, `pde_compound_data.npy`, and
  `ks_data.mat`, but it is not currently evaluated on systems such as
  Navier-Stokes.
- The shared-library setup makes the comparison mostly about structure
  selection and coefficient estimation, not about different candidate spaces.
  This is useful for fair benchmarking, but it also means framework-specific
  advantages such as DISCOVER MODE2/PINN denoising are not included in these
  numbers.
