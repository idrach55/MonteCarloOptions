# MonteCarloOptions
Monte Carlo simulation for options pricing in C and CUDA C++.
Both C and CUDA C++ programs can price vanilla european calls.

# CUDA C++ Instructions
Edit the source code to set contract parameters. Assuming the CUDA toolkit is installed from NVIDIA
(https://developer.nvidia.com/cuda-downloads), compile with

    nvcc -o mc.o montecarlo.cu
    
You may use the command line argument to set the number of blocks (each block runs 1024 trajectories).

    ./mc.o -b 100

Or do not include any arguments to use the default (200 blocks = 204,800 trajectories).

# C Instructions
Edit source code to set contract parameters. Then compile with

    gcc -o mc.o montecarlo.c
    
You may use the command line argument to set the number of trajectories.

    ./mc.o -T 30000
    
Or do not include any arguments to use the default (30k).
