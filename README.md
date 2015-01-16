# MonteCarloOptions
Monte Carlo simulation for options pricing in CUDA C++.
Currently supports vanilla european calls.

# Usage Instructions
Edit the source code to set contract parameters. Assuming the CUDA toolkit is installed from NVIDIA
(https://developer.nvidia.com/cuda-downloads), compile with

    nvcc -o mc.o montecarlo.cu
    
Command line arguments:<br>
<code>-b blocks</code> set total number of blocks (each runs 1024 trajectories), default is 200<br>
<code>-m max per partition</code> set maximum number of blocks per kernel call, default is 500<br>
<code>-N time steps</code> set number of time steps for each trajectory, default is 500

# Kernel Timeouts
There has been an issue with kernel timeouts. If the card running this program is also driving a display, kernels running for more than an OS specified amount of time will be killed. To counter this, I have written a block paritioning system to split
out the kernel calls over sequentially run smaller groupings. The default is a maximum of 500 blocks per group. Additionally, using more than 9000 timesteps may require fewer blocks per group. These parameters are calibrated to my GeForce GT 650M and may be raised or lowered with higher or lower performance cards respectively. Refer to the above command line arguments to adjust these parameters.
