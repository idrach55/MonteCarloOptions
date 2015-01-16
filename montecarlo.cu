//
//  montecarlo.cu
//  MonteCarloOptions
//
//  Created by Isaac Drachman on 1/15/15.
//  Copyright (c) 2015 Isaac Drachman. All rights reserved.
//

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// define struct for passing simulation parameters
typedef struct
{
    int N;
    double S0;
    double sigma;
    double r;
    double T;
    double K;
} params;

/*
 description:   initializes curand (CUDA rng)
 parameters:    
                curandState *state:    pointer to the random number generator
                unsigned int *seed:    value with which to seed rngs
 output:        none       
*/
__global__ void init_curand(curandState *state, unsigned int *seed) 
{
    // we seed an rng for each trajectory
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(*seed, idx, 0, &state[idx]);
}

/*
 description:   runs a single trajectory in the monte carlo simulation
 parameters:    
                curandState *state:    pointer to the random number generator
                params *p:             parameters for simulation
                double *payoffs:       payoff array to put result
 output:        none       
*/
__global__ void single_trajectory(curandState *state, params *p, double *payoffs) 
{
    // calculate our index (which trajectory is this)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // copy this thread's rng to local memory
    curandState lcl_state = state[idx];

    // set stepsize and initial price
    double dt = p->T / p->N;
    double S = p->S0;
    // for each time step 1 through N
    for (int n = 1; n <= p->N; n++) 
    {
        // using Euler-Murayama discretization for the geometric Bronwian model of stock dynamics
        // drift and diffuse the stock price over one timestep
        // taking advantage of curand's standard normal distribution draw function
        S *= (1 + p->r*dt + p->sigma*curand_normal(&lcl_state)*sqrt(dt));
    }
    // if this option makes money, we have a payoff, otherwise 0
    if (S - p->K > 0.0) payoffs[idx] = S - p->K;
    else payoffs[idx] = 0.0;
}

int main(int argc, char **argv) 
{
    // fill out simulation parameters to pass to GPU
    params h_params;
    h_params.N = 500;            // time steps
    h_params.S0 = 1992.67;       // spot price
    h_params.sigma = 0.17056;    // volatility (annualized)
    h_params.r = 0.00023;        // risk-free interest rate (annualized)
    h_params.T = 9.0/365.0;      // time to maturity in years
    h_params.K = 1990.00;        // strike price


    // default number of blocks is 200
    // each block runs 1024 threads (trajectories)
    int nBlocks = 200;
    int maxBlocks = 500;
    int nThreads = 1024;

    // check for command line arguments
    if (argc > 2) {
        for (int i = 1; i < argc; i++) 
        {
            if (strcmp(argv[i],"-b") == 0) nBlocks = atoi(argv[i+1]);
            if (strcmp(argv[i],"-m") == 0) maxBlocks = atoi(argv[i+1]);
            if (strcmp(argv[i],"-N") == 0) h_params.N = atoi(argv[i+1]);
        }
    } else if (argc == 2 && strcmp(argv[1],"--help") == 0) {
        std::cout << "usage: " << argv[0] << " [-b blocks] [-m max per partition] [-N time steps] [--help]" << std::endl;
        return -1;
    }

    // we will partition the blocks over several kernels to avoid timeout errors
    // each kernel will execute at most 500 blocks   
    int nPartitions = 1;
    if (nBlocks > maxBlocks && nBlocks % maxBlocks == 0) nPartitions = nBlocks / maxBlocks;
    else if (nBlocks > maxBlocks) nPartitions = nBlocks / maxBlocks + 1; 

    int partition[nPartitions];

    int tmp_nBlocks = nBlocks;
    for (int p = 0; tmp_nBlocks > 0; p++) {
        int amt = min(tmp_nBlocks,maxBlocks);
        partition[p] = amt; 

        tmp_nBlocks -= amt;
    }

    double total_sum = 0.0;
    // for each group of blocks
    for (int p = 0; p < nPartitions; p++) 
    {
        nBlocks = partition[p];

        // seed the random number generator
        // using time on the host system and passing it to the GPU
        unsigned int h_seed = (unsigned int)time(NULL);
        // make space in VRAM and copy over seed
        unsigned int *d_seed;
        cudaMalloc(&d_seed, sizeof(unsigned int));
        cudaMemcpy(&d_seed, &h_seed, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // setup rng
        curandState *d_state;
        // we'll have a seperate generator for each trajectory
        cudaMalloc(&d_state, nBlocks * nThreads);
        // initialize this generator with seed
        init_curand<<< nBlocks, nThreads >>>(d_state, d_seed);

        // make space in VRAM and copy over parameters
        params *d_params;
        cudaMalloc(&d_params, sizeof(params));
        cudaMemcpy(d_params, &h_params, sizeof(params), cudaMemcpyHostToDevice);

        // make space on device for payoff array
        double *d_payoffs;
        cudaMalloc(&d_payoffs, sizeof(double)*nBlocks*nThreads);

        // run our trajectories
        single_trajectory<<< nBlocks, nThreads >>>(d_state, d_params, d_payoffs);

        // dynamically allocate payoff array on host
        double *h_payoffs = new double[nBlocks*nThreads];
        // copy payoffs from device to host
        cudaMemcpy(h_payoffs, d_payoffs, sizeof(double)*nBlocks*nThreads, cudaMemcpyDeviceToHost);

        // sum each payoff
        double sum = 0.0;
        for (int m = 0; m < nBlocks*nThreads; m++) sum += h_payoffs[m];    

        // calculate average for this partition
        total_sum += sum / (nBlocks*nThreads);

        // free memory on host and device for this partition
        delete[] h_payoffs;
        cudaFree(d_state);
        cudaFree(d_params);
        cudaFree(d_payoffs);

        std::string error = cudaGetErrorString(cudaGetLastError());
        if (error.compare("the launch timed out and was terminated") == 0) 
        {
            std::cout << "error: kernel timeout" << std::endl;
            return 0;
        }
    }

    // calculate discounted average payoff and print
    double premium = exp(-h_params.r * h_params.T)*(total_sum/nPartitions);
    std::cout << "european call premium = " << premium << std::endl;

    // exit
    return 0;
}