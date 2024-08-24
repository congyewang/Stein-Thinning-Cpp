# Stein-Thinning-Cpp

The C++ version of Stein Thinning.

> `Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., Oates, C. J. (2022). Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4), 1059-1081`.

## Requirements

```bash
sudo apt update
sudo apt install build-essential
sudo apt install cmake

## Optional
sudo apt install ninja-build
sudo apt install libopenblas-dev
```


## Compilation

```bash
mkdir build && cd build
cmake ..
make

## Optional
cmake -G Ninja ..
ninja
```

## Utilisation

```bash
./stein_thinning <int number_of_points> <str /path/to/smp.csv> <str /path/to/scr.csv> [str /path/to/output.csv]

# For Example
./stein_thinning 40 "../demo/sample_chains/gmm/smp.csv" "../demo/sample_chains/gmm/scr.csv" "../demo/sample_chains/gmm/output.csv"
```

## Test

```bash
cd test && mkdir build && cd build
cmake ..
make

# Optional
cmake -G Ninja ..
ninja

./stein_thinning_test
```
