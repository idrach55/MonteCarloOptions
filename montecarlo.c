//
//  montecarlo.c
//  MonteCarloOptions
//
//  Created by Isaac Drachman on 1/14/15.
//  Copyright (c) 2015 Isaac Drachman. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 description:	Box Muller implementation for drawing from the standard normal distribution
 parameters:	none
 output:		a draw from normal distribution with mean=0, sd=1
*/
double BoxMuller()
{
	// this is an implementation of the Box Muller method
	double x1;
	double x2;
	double square;
	do {
		// draw two numbers from a uniform distribution between [-1,1]
		x1 = 2.0*(double)(rand())/(double)(RAND_MAX)-1.0;
		x2 = 2.0*(double)(rand())/(double)(RAND_MAX)-1.0;
		// sum squares
		square = x1*x1 + x2*x2;
	} while (square >= 1.0);
	return x1*sqrt(-2.0*log(square)/square);
}

/*
 description:	monte carlo simulation for pricing a vanilla European Call
 parameters:
				int M:			number of trajectories to simulate stock price
				int N:			number of timesteps in discretization
				double S0:		spot price (at time t=0)
				double sigma:	stock volatility (annualized)
				double r:		risk-free interest rate (annualized)
				double T:		time to maturity in years
				double K:		contract strike price
 output:		option premium, approximated by simulation's discounted average payoff
*/
double EuropeanCall(int M, int N, double S0, double sigma, double r, double T, double K)
{	
	double dt = T / N;
	double sum = 0.0;
	double S;
	
	// for each trajectory
	for (int m = 0; m < M; m++) {
		// initialize trajectory
		S = S0;
		// for each timestep 1 through N
		for (int n = 1; n <= N; n++) {
			// using Euler-Murayama discretization for the geometric Bronwian model of stock dynamics
			// drift and diffuse the stock price over one timestep
			S *= 1 + r*dt + sigma*BoxMuller()*sqrt(dt);
		}
		// if this option makes money, we have a payoff, otherwise 0
		if (S-K > 0.0) sum += S-K;
	}
	// discount the average payoff and return
	return exp(-r*T)*(sum/M);
}

int main(int argc, const char * argv[]) {
	// seed the random number generator
	srand((unsigned int)time(NULL));
	
	int M = 80000;				// trajectories
	int N = 500;				// timesteps
	double S0 = 109.33;			// spot price
	double sigma = 0.2519;		// volatility (annualized)
	double r = 0.0025;			// risk-free interest rate (annualized)
	double T = 48.0/365.0;		// time to maturity in years
	double K = 120.00;			// strike price
	
	// calculate premium for a European Call with above parameters
	double premium = EuropeanCall(M, N, S0, sigma, r, T, K);
	
	// print values and quit
	printf("european call premium = %0.6f\n",premium);
	return 0;
}
