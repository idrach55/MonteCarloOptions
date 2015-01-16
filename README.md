# MonteCarloOptions
Monte Carlo simulation for options pricing in CUDA C++.
Currently supports vanilla european calls.

# Usage Instructions
Edit the source code to set contract parameters. Assuming the CUDA toolkit is installed from NVIDIA
(https://developer.nvidia.com/cuda-downloads), compile with

    nvcc -o mc.o montecarlo.cu
    
You may use the command line argument <code>-b blocks</code> to set the number of blocks (each block runs 1024 trajectories).
The default is 200 blocks = 204,800 trajectories.

# Kernel Timeouts
There has been an issue with kernel timeouts. If the card running this program is also driving a display, kernels running for more than an OS specified amount of time will be killed. To counter this, I have written a block paritioning system to split
out the kernel calls over sequentially run smaller groupings. The default is a maximum of 500 blocks per group. Additionally, using more than 9000 timesteps may require fewer blocks per group. These parameters are calibrated to my GeForce GT 650M and will may be raised or lowered with higher or lower performance cards respectively.
