//
// Created by hao on 2019-12-12.
//

// own header
#include "../header/candidateSelection.h"

// user defined headers
#include "../header/stdafx.h"
#include "../header/utils.h"

// standard libraries
#include <boost/math/special_functions/beta.hpp>
#include <math.h>
#include <random>
#include <vector>

using std::vector;
using std::cout;
using std::endl;
using namespace utils;

namespace candidateSelection {

	/*
	* First choice hill climbing
	* finds theta that maximises PDIS(D_c,theta,pi_b)
	* with constraint (refer to CS687 notes)
	*/
	// TODO: test this function
	vector<float> fchc_candidate_selection(const vector<vector<float>> &dataCandidate, vector<float> &init_theta, const vector<float> &theta_b, const Fchc_params &params) {
		vector<float> theta_c; // candidate policy params
		vector<float> theta_sample;
		float curr_best_pdis = 0; // current best pdis

								  // unpack params
		int iter = params.iter; // number of iterations to run the fchc algo
		float mean = params.mean; // mean of the gaussian
		float stddev = params.stddev; // stddev of the gaussian
		bool useSeed = params.useSeed; // flag for using seed. 0 if not using seed. 1 if using seed.
		int seed = params.seed; // seed for random number generator
		int m = params.m; // dim of state features
		int k = params.k; // order of fourier basis
		int numAct = params.numAct; // num of possible actions
		double delta = params.delta; // confidence bound
		int safety_size = params.safety_size; // size of safety dataset
		float c = params.c; // lower bound for safety test

		std::default_random_engine generator;
		if (useSeed) {
			generator.seed(seed); // seed the random number generator
		}
		std::normal_distribution<double> distribution(mean, stddev); // gaussian

		theta_c = init_theta;
		for (int i = 0; i<iter; i++) {
			cout << "In FCHC iter: " << i << endl;
			// get sample policy theta_sample by jittering init_theta with a gaussian
			theta_sample.clear(); // clear vector to size 0
			for (int j = 0; j<theta_c.size(); j++) {
				double number = distribution(generator);
				theta_sample.push_back(theta_c[j] + number); // gaussian noise
			}

			// make sure theta_sample satisfies the constraint
			float first_term = PDIS_avg(dataCandidate, theta_sample, theta_b, m, k, numAct);
			float second_term = 2 * PDIS_stddev(dataCandidate, theta_sample, theta_b, m, k, numAct)*float(tinv(delta, safety_size)) / sqrt(safety_size);
			if (first_term - second_term < c) { // if constraint is not met
												//continue; // this theta_sample is thrown away
				break;
			}

			// now that we have satisfied the constraint,
			// we evaluate theta_sample. calculate PDIS(D_c, theta_sample, pi_b)
			float pdis_sample = PDIS_avg(dataCandidate, theta_sample, theta_b, m, k, numAct);

			cout << "got better pdis" << endl;

			if (pdis_sample > curr_best_pdis) {
				curr_best_pdis = pdis_sample; // best pdis so far
				theta_c = theta_sample; // best theta so far
			}
		}

		if (theta_c.size() != 0) {
			cout << "Candidate Selection Done!" << endl;
			cout << "PDIS of candidate: " << curr_best_pdis << endl;
			cout << "Theta_c: ";
			for (int j = 0; j<theta_c.size(); j++) {
				cout << theta_c[j] << "\t";
			}
			cout << endl;
		}

		return theta_c;
	}

} // end namespace candidateSelection

