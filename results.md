# Current Benchmark Results

These are the current PySINDy and DeepMoD benchmark results used for comparison.
For multi-equation systems, only the aggregate `system` row is shown.

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

| Dataset | Target | Time, s | Library | RE sum |
|---|---:|---:|---:|---:|
| ode_data.npy | u_tt | 2.637 | 12 | 0.0165407 |
| vdp_data.npy | u_tt | 0.09897 | 12 | 0.0880973 |
| lotka_data.npy | system | 0.1894 | 20 | 0.0106256 |
| lorenz_data.npy | system | 0.3687 | 60 | 0.00229907 |
| burgers_data.mat | u_t | 0.5387 | 33 | 0.0156706 |
| burgers_sln_100_data.csv | u_t | 0.327 | 33 | 0.000215411 |
| ac_data.npy | u_t | 0.588 | 33 | 0.0136804 |
| kdv_data.mat | u_t | 1.363 | 33 | 0.0331566 |
| kdv_periodic_data.npy | u_t | 0.9184 | 33 | 0.00646051 |
| wave_data.csv | u_tt | 0.5317 | 33 | 0.00295097 |
| pde_compound_data.npy | u_t | 0.7048 | 33 | 0.00036782 |
| pde_divide_data.npy | u_t | 3.346 | 33 | 0.000363708 |
| ks_data.mat | u_t | 3.297 | 33 | 0.036629 |
| ns_data.mat | system | 6.066 | 23 | 0.435144 |
| ODE_simple_discovery | u_t | 0.07996 | 11 | 0.000755457 |

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
| ODE_simple_discovery | u_t | 10 | 3/30 | 3.7 | 2.231 | 0.04757 | 0.0178 |

### DeepMoD

| Dataset | Target | Noise level | Correct | HD mean | HD std | RE mean | RE std |
|---|---:|---:|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | 2.7 | 4/30 | 1.633 | 1.189 | 0.6472 | 0.4069 |
| vdp_data.npy | u_tt | 0.15 | 3/30 | 1.767 | 0.6261 | 0.1655 | 0.05422 |
| lorenz_data.npy | system | 0.01 | 3/30 | 0.9 | 0.3051 | 0.05281 | 0.07562 |
| lotka_data.npy | system | 9 | 3/30 | 2.467 | 1.634 | 0.1677 | 0.03882 |
| burgers_data.mat | u_t | 0.65 | 3/30 | 1.3 | 0.7497 | 0.03259 | 0.01482 |
| ac_data.npy | u_t | 15 | 5/30 | 3.533 | 1.833 | 0.8475 | 0.2111 |
| kdv_data.mat | u_t | 0.0085 | 4/30 | 2.6 | 1.037 | 0.2384 | 0.01568 |
| kdv_periodic_data.npy | u_t | 2.4e-05 | 4/30 | 10.3 | 6.859 | 0.005094 | 0.0005008 |
| wave_data.csv | u_tt | 0.0195 | 3/30 | 1.7 | 1.489 | 0.0372 | 0.03729 |
| pde_divide_data.npy | u_t | 0.002 | 4/30 | 17.53 | 8.08 | 0.0003107 | 6.12e-05 |
| pde_compound_data.npy | u_t | 0.04 | 3/30 | 5.933 | 4.315 | 0.02843 | 0.002302 |
| ns_data.mat | system | 0.355 | 4/30 | 1.367 | 0.8087 | 0.3429 | 0.004472 |
| ks_data.mat | u_t | 0.008 | 4/30 | 1.4 | 1.07 | 2.422 | 0.02156 |
| burgers_sln_100_data.csv | u_t | 1.5 | 3/30 | 1.767 | 1.104 | 0.07697 | 0.004783 |
| ODE_simple_discovery | u_t | 14 | 3/30 | 1 | 0.4549 | 0.2251 | 0.2023 |
