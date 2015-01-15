//
//  main.c
//  MonteCarloOptions
//
//  Created by Isaac Drachman on 1/14/15.
//  Copyright (c) 2015 Isaac Drachman. All rights reserved.
//

#include <stdio.h>
#include "montecarlo.h"

int main(int argc, const char * argv[]) {
	int M = 20000;				// trajectories
	int N = 1000;				// timesteps
	double S0 = 109.33;			// spot price
	double sigma = 0.2519;		// volatility (annualized)
	double r = 0.0025;			// risk-free interest rate (annualized)
	double T = 48.0/365.0;		// time to maturity in years
	double K = 120.00;			// strike price
	
	// calculate premium for a European Call with above parameters
	double premium = EuropeanCall(M, N, S0, sigma, r, T, K);
	
	// print value and quit
	printf("european call premium = %0.6f\n",premium);
    return 0;
}
