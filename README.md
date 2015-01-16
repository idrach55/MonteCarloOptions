# MonteCarloOptions
Monte Carlo simulation for options pricing in C and CUDA C++.
Both C and CUDA C++ programs can price vanilla european calls.

Usage for CUDA C++:<br>
Edit source code to set contract parameters. Assuming the CUDA toolkit is installed from NVIDIA
(https://developer.nvidia.com/cuda-downloads), compile with<br>
    <code>nvcc -o mc.o montecarlo.cu</code><br>
You may use the command line argument to set the number of blocks (each block runs 1024 trajectories).
    <code>./mc.o -b 100</code><br>
Or do not include any arguments to use the default (200 blocks = 204,800 trajectories).

Usage for C:
Edit source code to set contract parameters. Then compile with
    gcc -o mc.o montecarlo.c
You may use the command line argument to set the number of trajectories.
    ./mc.o -T 30000
Or do not include any arguments to use the default (30k).
