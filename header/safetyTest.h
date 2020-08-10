//
// Created by hao on 2019-12-11.
//

#ifndef CS687_FINAL_PROJECT_SAFETYTEST_H
#define CS687_FINAL_PROJECT_SAFETYTEST_H

#include "stdafx.h"

#include <vector>

using std::vector;

namespace SafetyTest {

	/*
	* safety test for high confidence policy improvement
	* Uses student's T-test
	*
	* B = PDIS(D_s, pi_e, pi_b) - (stddev/sqrt(size_ds))*tinv
	* c = the lower bound for the expected return of pi_e, J(pi_e).
	* We can set c to to J(pi_b), the expected return of the behavior policy.
	*
	* Input:
	*     safetyData      : safety dataset. vector of histories
	*     theta_c         : candidate policy params
	*     theta_b         : behavior policy params
	*     c               : lower bound for J(pi_e)
	*     m               : dim of state features (before applying phi)
	*     k               : order of fourier basis
	*     numAct          : num of possible actions
	*     delta           : confidence bound
	*
	* Output:
	*     isSafe          : 1 if candidate policy is safe. 0 if not.
	*
	* Algo:
	*     If B >= c, return 1.
	*     Else, return 0 (no solution found).
	*/
	int safetyTest(const vector<vector<float>> &safetyData, const vector<float> &theta_c, const vector<float> &theta_b, float c, int m, int k, int numAct, float delta);

} // end namespace safetyTest


#endif //CS687_FINAL_PROJECT_SAFETYTEST_H
