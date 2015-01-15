//
//  montecarlo.h
//  MonteCarloOptions
//
//  Created by Isaac Drachman on 1/14/15.
//  Copyright (c) 2015 Isaac Drachman. All rights reserved.
//

#ifndef MonteCarloOptions_montecarlo_h
#define MonteCarloOptions_montecarlo_h

// define functions implemented in montecarlo.c
double BoxMuller();
double EuropeanCall(int M, int N, double S0, double sigma, double r, double T, double K);

#endif
