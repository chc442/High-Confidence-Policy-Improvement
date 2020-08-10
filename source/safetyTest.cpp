//
// Created by hao on 2019-12-11.
//


// own header
#include "../header/safetyTest.h"

// user defined headers
#include "../header/stdafx.h"
#include "../header/utils.h"

// standard libraries
#include <boost/math/special_functions/beta.hpp>
#include <math.h>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace utils;

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
	// TODO: test this function
	int safetyTest(const vector<vector<float>> &safetyData, const vector<float> &theta_c, const vector<float> &theta_b, float c, int m, int k, int numAct, float delta) {
		int isSafe = 0; // flag indicating whether candidate policy is safe. 1=safe. 0=unsafe.
		int n_safety = safetyData.size();
		float B = 0;

		cout << "Starting Safety Test..." << endl;

		// calculate B
		float first_term = PDIS_avg(safetyData, theta_c, theta_b, m, k, numAct);
		float second_term = PDIS_stddev(safetyData, theta_c, theta_b, m, k, numAct) * float(tinv(delta, n_safety - 1)) / sqrt(n_safety);
		B = first_term - second_term;

		if (B >= c) {
			isSafe = 1; // candidate policy is safe
		}
		else {
			isSafe = 0;
		}

		cout << "Safety Test Done!" << endl;
		return isSafe;
	}

} // end namespace safetyTest
