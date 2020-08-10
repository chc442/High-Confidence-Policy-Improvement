//
// Created by hao on 2019-12-11.
//

#ifndef CS687_FINAL_PROJECT_UTILS_H
#define CS687_FINAL_PROJECT_UTILS_H

#include "stdafx.h"

#include <boost/math/special_functions/beta.hpp>
#include <vector>

using std::vector;

namespace utils {
	/*
	* This file contains:
	* the function for calculating phi(s)
	* the function for calculating pi(s,a)
	* the function for calculating PDIS
	* etc
	*/


	/*
	* Calculates the expected return of behavior policy, given the dataset
	* Assumes gamma = 1
	*/
	float baselineReturn(const vector<vector<float>> &dataset, int m);


	/*
	* Calculates phi(s)
	* Inputs:
	*          s           : state vector of dimension m (normalized)
	*          m           : dimension of state vector s
	*          k           : order of fourier basis
	* Outputs:
	*         phi          : vector of dimension (k+1)^m
	*/
	vector<float> calculatePhi(const vector<float> &s, int m, int k);


	/*
	* Calculates pi(s,a) given policy theta
	* Uses softmax.
	* Inputs:
	*           theta          : policy params
	*           s              : normalised state features (before applying phi)
	*           a              : action (int)
	*           m              : dimension of state features
	*           k              : order of fourier basis
	*           numActions     : total num of actions
	* Outputs:
	*           pi(s,a)        : probability that action a is taken in state s.
	*/
	float policy_prob(const vector<float> &theta, const vector<float> &s, int a, int m, int k, int numActions);


	/* Per-Decision Importance Sampling for a single history H
	* PDIS is an unbiased estimate of J(pi_e)
	* gamma: discount factor. set to 1.
	*
	* Inputs:
	*             hist         : one history
	*             theta_e      : evaluation policy params
	*             theta_b      : behavior policy params
	*             m            : dimension of state features (before applying phi)
	*             k            : order of fourier basis
	*             numAct       : num of possible actions
	* Outputs:
	*             PDIS         : PDIS for this one history
	*/
	float PDIS_single(const vector<float> &hist, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct);


	/*
	* Calculates PDIS on a whole dataset
	* This is an unbiased estimate of J(pi_e).
	* Takes the average of PDIS on each history
	* Inputs:
	*            dataset           : whole dataset. vector of histories
	*            theta_e           : evaluation policy params
	*            theta_b           : behavior policy params
	*            m                 : dimension of state features (before applying phi)
	*            k                 : order of fourier basis
	*            numAct            : num of possible actions
	* Outputs:
	*            pdis_avg          : PDIS applied to the whole dataset
	*
	*/
	float PDIS_avg(const vector<vector<float>> &dataset, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct);


	/*
	* Calculate sample std dev of PDIS on dataset
	* Inputs:
	*            dataset           : vector of histories
	*            theta_e           : evaluation policy params
	*            theta_b           : behavior policy params
	*            m                 : dimension of state features before applying phi
	*            k                 : order of fourier basis
	*            numAct            : num of possible actions
	* Outputs:
	*            pdis_stddev       : sample std dev of PDIS on this dataset
	*/
	float PDIS_stddev(const vector<vector<float>> &dataset, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct);


	/*
	* Returns the 100(1-p) percentile of the Student t distribution with nu degrees of freedom.
	* This is a C++ implementation of Matlab's tinv function.
	* Taken from:
	* https://aisafety.cs.umass.edu/tutorial4cpp.html
	*/
	double tinv(double p, int nu);

} // end namespace utils


#endif //CS687_FINAL_PROJECT_UTILS_H
