//
// Created by hao on 2019-12-10.
//

#ifndef CS687_FINAL_PROJECT_DATASPLIT_H
#define CS687_FINAL_PROJECT_DATASPLIT_H

#include "stdafx.h"

#include <vector>

using std::vector;

namespace DataSplit {

	/*
	* Splits the dataset into candidate/safety datasets, based on percent_candidate and percent_safety
	* percent_candidate and percent_safety should be a float in [0,1] and should add up to 1.
	*/
	void dataSplit(const vector<vector<float>> histories, vector<vector<float>> &data_candidate, vector<vector<float>> &data_safety, float percent_candidate, float percent_safety);

} // end namespace dataSplit


#endif //CS687_FINAL_PROJECT_DATASPLIT_H
