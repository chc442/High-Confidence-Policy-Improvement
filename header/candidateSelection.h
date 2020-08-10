//
// Created by hao on 2019-12-12.
//

#ifndef CS687_FINAL_PROJECT_CANDIDATESELECTION_H
#define CS687_FINAL_PROJECT_CANDIDATESELECTION_H


#include "stdafx.h"

#include <vector>

using std::vector;

namespace candidateSelection {

	struct Fchc_params {
		int iter;                   // number of iterations to run for
		float mean;                 // mean of the gaussian
		float stddev;               // stddev of the gaussian
		bool useSeed;               // flag for using seed. 0 if not using seed. 1 if using seed.
		int seed;                   // seed for rng
		int m;                      // dim of state features
		int k;                      // order of fourier basis
		int numAct;                 // num of possible actions
		double delta;                // confidence bound
		int safety_size;            // size of safety dataset
		float c;                    // lower bound for safety test
	};

	/*
	* First choice hill climbing
	* finds theta that maximises PDIS(D_c,theta,pi_b)
	* with constraint (refer to CS687 notes)
	*/
	vector<float> fchc_candidate_selection(const vector<vector<float>> &dataCandidate, vector<float> &init_theta, const vector<float> &theta_b, const Fchc_params &params);

} // end namespace candidateSelection


#endif //CS687_FINAL_PROJECT_CANDIDATESELECTION_H


