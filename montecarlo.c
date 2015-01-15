//
//  montecarlo.c
//  MonteCarloOptions
//
//  Created by Isaac Drachman on 1/14/15.
//  Copyright (c) 2015 Isaac Drachman. All rights reserved.
//

#include "montecarlo.h"
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
 description:	monte carlo simulation for pricing vanilla a European Call
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
	// seed the random number generator
	srand((unsigned int)time(NULL));
	
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
