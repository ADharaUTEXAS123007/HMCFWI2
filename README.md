# HMC Full waveform inversion

Framework for probabilistic Full Waveform Inversion using Hamiltonian Monte Carlo. Uses the specific 2D elastic wave 
equation and Virieux's 1986 implementation.

## Compilation

Make sure you have an OpenMP enabled compiler, otherwise this code will be too slow.

from the `bin/` directory:

```
g++ ../src/sampling_main.cpp ../src/sampler/hmc_sampler.cpp ../ext/forward-virieux/src/fdWaveModel.cpp -fopenmp -O3  -std=c++11 -o main-executable
```

## Execution

Run from the `bin/` directory the following commnads:

```
export OMP_NUM_THREADS=6           
./main-executable
```


## Configuring the sampling and finite difference modelling

All the necessary settings for HMC and the FD model can be found in the following two files in the `bin/` directory:
```
configuration_structural_target_fd.ini
configuration_structural_target_hmc.ini

```

If you change the name of the first file (`configuration_structural_target_fd.ini`) make sure this is updated in the second (`configuration_structural_target_hmc.ini`). If you change the name of the second, make sure it is updated in the main .cpp file (`src/sampling_main.cpp`).
