# Stein-Thinning-Cpp

C++ version of Stein Thinning.

# Compilation

```bash
mkdir build && cd build
cmake ..
make
```

# Utilisation

```bash
./stein_thinning <int number_of_points> <str path_to_smp.csv> <str path_to_scr.csv>

# For Example
# ./stein_thinning 40 "../demo/sample_chains/gmm/smp.csv" "../demo/sample_chains/gmm/scr.csv"
```

# Test

```bash
cd test && mkdir build && cd build
cmake ..
make
```
