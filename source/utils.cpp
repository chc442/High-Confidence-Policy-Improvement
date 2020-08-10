//
// Created by hao on 2019-12-11.
//

#include "../header/utils.h"

#include "../header/stdafx.h"

// standard libraries
#include <boost/math/special_functions/beta.hpp>
#include <math.h>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::cout;
using std::endl;
using std::vector;
using std::inner_product;


namespace utils {

	/*
	* Calculates the expected return of behavior policy, given the dataset
	* Assumes gamma = 1
	*/
	float baselineReturn(const vector<vector<float>> &dataset, int m) {
		float all_returns = 0; // sum of returns of all histories
		for (int i = 0; i<dataset.size(); i++) { // for each history
			vector<float> hist = dataset[i]; // history at index i
			int L = hist.size() / (m + 2); // horizon of this history
			float curr_return = 0; // return for this history
			for (int t = 0; t<L; t++) {
				float reward = hist[t*(m + 2) + (m + 1)]; // reward at time t
				curr_return += reward;
			}
			all_returns += curr_return;
		}
		float exp_return = all_returns / dataset.size();
		return exp_return;
	}


	/*
	* Calculates phi(s)
	* Inputs:
	*          s           : state vector of dimension m (normalized)
	*          m           : dimension of state vector s
	*          k           : order of fourier basis
	* Outputs:
	*         phi          : vector of dimension (k+1)^m
	*/
	vector<float> calculatePhi(const vector<float> &s, int m, int k) {
		int phi_size = pow((k + 1), m); // dimension of phi is (k+1)^m
		vector<float> phi;
		vector<float> c_i;
		// initialise c_i to be a vector of zeros. of dimension m
		for (int i = 0; i<m; i++) {
			c_i.push_back(0);
		}

		// calculate each element in phi
		for (int i = 0; i< phi_size; i++) {
			float dot_prod = 0;
			dot_prod = inner_product(std::begin(c_i), std::end(c_i), std::begin(s), 0.0); // dot(c_i, s)
			float phi_i = cos(M_PI * dot_prod);
			phi.push_back(phi_i);

			if (i<phi_size - 1) {
				// increment c_i (little endian, base k+1)
				int idx = 0;
				c_i[idx] += 1;
				// propagate carry overs
				while (c_i[idx] == (k + 1)) {
					c_i[idx] = 0;
					c_i[idx + 1] += 1;
					idx++;
				}
			}
		}
		return phi;
	}


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
	float policy_prob(const vector<float> &theta, const vector<float> &s, int a, int m, int k, int numActions) {
		float nominator = 0;
		float denominator = 0;
		float probability = 0; // pi(s,a). using policy theta
		vector<float> phi_s; // dimension (k+1)^m
		vector<float> theta_action; // theta values corresponding to an action. dimension (k+1)^m
		float dot_prod = 0;
		int l = pow((k + 1), m); // size of theta_action. (k+1)^m

								 // calculate phi(s)
		phi_s = calculatePhi(s, m, k);

		// find theta for action a
		theta_action = vector<float>(theta.begin() + a * l, theta.begin() + (a + 1)*l);

		dot_prod = inner_product(begin(phi_s), end(phi_s), begin(theta_action), 0.0); // dot(phi_s,theta_action)
		nominator = exp(dot_prod);

		for (int action = 0; action<numActions; action++) {
			theta_action = vector<float>(theta.begin() + action * l, theta.begin() + (action + 1)*l);
			dot_prod = inner_product(begin(phi_s), end(phi_s), begin(theta_action), 0.0); // dot(phi_s,theta_action)
			denominator += exp(dot_prod);
		}

		probability = nominator / denominator;
		return probability;
	}


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
	// TODO: test this function
	float PDIS_single(const vector<float> &hist, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct) {
		int L = hist.size() / (m + 2); // horizon of episode
		float product_term = 1; // cache for product term inside the sum
		float pdis = 0;
		vector<float> curr_state; // state features at time t. dimension m
		int curr_act; // action at time t

		for (int t = 0; t<L; t++) {
			// in this for loop we calculate each term corresponding to time t, and sum all of them
			float gamma_term = 1;  // using gamma 1
			float reward_term = hist[t*(m + 2) + (m + 1)]; // reward at time t

			curr_state = vector<float>(hist.begin() + t * (m + 2), hist.begin() + t * (m + 2) + m); // state features at time t
			curr_act = hist[t*(m + 2) + m]; // action taken at time t
			product_term = product_term * (policy_prob(theta_e, curr_state, curr_act, m, k, numAct) / policy_prob(theta_b, curr_state, curr_act, m, k, numAct));

			pdis += (gamma_term * product_term * reward_term); // sum over all t
		}

		return pdis;
	}

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
	*/
	// TODO: test this function
	float PDIS_avg(const vector<vector<float>> &dataset, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct) {
		float pdis_avg = 0; // PDIS applied to the whole dataset
		float pdis_sum = 0; // sum of PDIS of all histories
		int num_episodes = dataset.size(); // number of episodes in the dataset

		for (int i = 0; i<num_episodes; i++) {
			vector<float> hist = dataset[i]; // history indexed at i
			float pdis = PDIS_single(hist, theta_e, theta_b, m, k, numAct); // PDIS for this single history
			pdis_sum += pdis;
		}
		pdis_avg = pdis_sum / float(num_episodes);
		return pdis_avg;
	}


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
	// TODO: test this func
	float PDIS_stddev(const vector<vector<float>> &dataset, const vector<float> &theta_e, const vector<float> &theta_b, int m, int k, int numAct) {
		float pdis_avg = PDIS_avg(dataset, theta_e, theta_b, m, k, numAct);
		int num_episodes = dataset.size(); // number of episodes in dataset
		float pdis_stddev = 0; // sample std dev of PDIS on dataset
		float sum = 0;

		// calculate summand indexed at i
		for (int i = 0; i<num_episodes; i++) {
			vector<float> hist = dataset[i]; // history at index i
			float curr_pdis = PDIS_single(hist, theta_e, theta_b, m, k, numAct); // pdis for this history
			sum += pow((curr_pdis - pdis_avg), 2);
		}
		pdis_stddev = sqrt(sum / float(num_episodes - 1));
		return pdis_stddev;
	}


	/*
	* Returns the 100(1-p) percentile of the Student t distribution with nu degrees of freedom.
	* This is a C++ implementation of Matlab's tinv function.
	* Taken from:
	* https://aisafety.cs.umass.edu/tutorial4cpp.html
	*/
	double tinv(double p, int nu)
	{
		// See "quantile" block here: https://www.boost.org/doc/libs/1_58_0/libs/math/doc/html/math_toolkit/dist_ref/dists/students_t_dist.html
		int v = nu; // To match Boost's documentation
		double x = boost::math::ibeta_inv(v / 2.0, 0.5, 2.0 * std::min(p, 1.0 - p));
		double y = 1.0 - x;
		return boost::math::sign(p - 0.5) * sqrt(v * y / x);
	}

} // end namespace utils
