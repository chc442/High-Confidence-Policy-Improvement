//
// Created by hao on 2019-12-10.
//

#include "../header/dataSplit.h""

#include "../header/stdafx.h"

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

namespace DataSplit {

	/*
	* Splits the dataset into candidate/safety datasets, based on percent_candidate and percent_safety
	* percent_candidate and percent_safety should be a float in [0,1] and should add up to 1.

	* TODO:
	*     use only percent_candidate. no need for percent_safety
	*/
	void dataSplit(const vector<vector<float>> histories, vector<vector<float>>&data_candidate, vector<vector<float>> &data_safety, float percent_candidate, float percent_safety) {
		cout << endl;
		cout << "Splitting Dataset into " << percent_candidate << "/" << percent_safety << " Candidate/Safety split" << endl;

		// sequential split
		int num_histories_total = histories.size();
		int num_histories_cand = int(num_histories_total * percent_candidate);
		//cout << "Candidate Data has " << num_histories_cand << " episodes" << endl;
		int num_histories_safety = num_histories_total - num_histories_cand;
		//cout << "Safety Data has " << num_histories_safety << " episodes" << endl;

		// construct data_candidate
		std::copy(histories.begin(), histories.begin() + num_histories_cand,
			std::back_inserter(data_candidate));

		// construct safety data
		std::copy(histories.begin() + num_histories_cand, histories.end(),
			std::back_inserter(data_safety));

	}

} // end namespace dataSplit




