/*
 * CS 687 Fall 2019 Final Project
 * High Confidence Policy Improvement
 * Author: Hao Cheam
 */

// user-defined headers
#include "../header/candidateSelection.h"
#include "../header/dataSplit.h"
#include "../header/safetyTest.h"
#include "../header/stdafx.h"
#include "../header/utils.h"

// standard libraries
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>


using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace candidateSelection;
using namespace DataSplit;
using namespace SafetyTest;
using namespace utils;


// CSV parsing code taken from:
// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
class CSVRow
{
public:
    std::string const& operator[](std::size_t index) const
    {
        return m_data[index];
    }
    std::size_t size() const
    {
        return m_data.size();
    }
    void readNextRow(std::istream& str)
    {
        std::string         line;
        std::getline(str, line);

        std::stringstream   lineStream(line);
        std::string         cell;

        m_data.clear();
        while(std::getline(lineStream, cell, ','))
        {
            m_data.push_back(cell);
        }
        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty())
        {
            // If there was a trailing comma then add an empty element.
            m_data.push_back("");
        }
    }
private:
    std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}


// main entry point of program
// TODO: modularise this file
int main()
{
    int num_of_trials = 100; // number of safe policies to produce. will create csv files
    float delta = 0.05;   // confidence bound
    int m=0; // dimension of state features
    int numActions=0; // number of actions
    int k=0; // fourier basis used by behavior policy
    vector<float> theta_b; // theta for behavior policy
    int n=0; // number of histories (episodes) in dataset
    vector<vector<float>> histories; // histories (S,A,R,...). histories[0] is first history, etc.

    // parse the CSV file
	int i = 0;
	std::ifstream       file("../data.csv");
	CSVRow              row;
    while(file >> row)
    {
        if(i==0){
            // parsing m
            m = stoi(row[0]); // dimension of state features
            cout << "m = " << m << "\n";
        }
        else if(i==1){
            // parsing numActions
            numActions = stoi(row[0]); // number of actions
            cout << "numActions = " << numActions << "\n";
        }
        else if(i==2){
            // parsing k
            k = stoi(row[0]); // fourier basis used by behavior policy
            cout << "Fourier Basis Order = " << k << "\n";
        }
        else if(i==3){
            // parsing theta for behavior policy
            cout << "Theta_b: ";
            for(int j=0; j<row.size(); j++){
                theta_b.push_back(stof(row[j]));
                cout << stof(row[j]) << "\t"; // note: stof is needed
            }
            cout << endl;
        }
        else if(i==4){
            // parsing n
            n = stoi(row[0]); // number of episodes in dataset
            cout << "Num Episodes = " << n << "\n";
        }
        else{
            // parsing history (S,A,R)
            // S is represented by m real numbers.
            //cout << "Episode " << (i-5) << ": ";
            vector<float> curr_hist;
            for(int j=0;j<row.size();j++){
                curr_hist.push_back(stof(row[j]));
                //cout << stof(row[j]) << "\t";
            }
            histories.push_back(curr_hist);
            //cout << endl;
        }

        i++;  
    }

    // throw away last row of data
    histories.pop_back();

    // print length of histories to make sure it is equal to n
    // cout << "Num of Histories: " << histories.size() << endl;

    // split dataset into candidate data and safety data
    vector<vector<float>> data_candidate; // candidate data
    vector<vector<float>> data_safety; // safety data
    float percent_candidate = 0.6; // percentage of dataset allocated to candidate data
    float percent_safety = float(1.0) - percent_candidate; // percentage of dataset allocated to safety data
    dataSplit(histories, data_candidate, data_safety, percent_candidate, percent_safety);

    // make sure the split is correct
    cout << "Size of Candidate Data: " << data_candidate.size() << endl;
    cout << "Size of Safety Data: " << data_safety.size() << endl;

    float return_b = baselineReturn(histories,m); // expected return of behavior policy, given dataset
    cout << "Return of Behavior Policy: " << return_b << endl;

    bool isDone = false; // false if we want to keep producing safe policies. true when we have created enough safe policies
    int policy_num = 1; // label for the current safe policy
    cout << "Creating " <<num_of_trials<< " safe policies..." << endl;
    std::default_random_engine generator; // this has to be outside the while loop
    // we stop when we have produced [num_of_trials] safe policies
    while(isDone==false){
        // using candidate data, we do candidate selection
        vector<float> init_theta;
        // generator.seed(10); // COMMENT THIS OUT after testing
        std::normal_distribution<double> distribution(0, 5); // gaussian
        // get init theta by jittering theta_b with a gaussian
        for(int j=0; j<theta_b.size(); j++){
            double number = distribution(generator);
            init_theta.push_back(theta_b[j] + number); // gaussian noise
        }

        Fchc_params params = {
                10,                   // number of iterations to run for
                0,                 // mean of the gaussian
                0.5,               // stddev of the gaussian
                0,               // flag for using seed. 0 if not using seed. 1 if using seed.
                0,                   // seed for rng
                m,                      // dim of state features
                k,                      // order of fourier basis
                numActions,                 // num of possible actions
                delta,                // confidence bound delta
                int(data_safety.size()),            // size of safety dataset
                return_b,                    // lower bound for safety test, c
        };

        // print init theta
        cout << "Running FCHC with theta_init: ";
        for(int j=0; j<init_theta.size(); j++){
            cout << init_theta[j] << "\t";
        }
        cout << endl;

        vector<float> theta_c = fchc_candidate_selection(data_candidate,init_theta, theta_b, params);
        if(theta_c.size()==0){ // this happens when fchc cannot find a better solution within the number of iterations
            //continue; // we skip the safety test and do another candidate selection
            theta_c = init_theta;
        }


        // using safety data, we do safety test on the candidate policy theta
        float c = params.c; // lower bound for safety test, c
        int isSafe = safetyTest(data_safety,theta_c,theta_b,c,m,k,numActions,delta);

        if(isSafe){
            cout << "Candidate Policy is Safe!" << endl;

            cout << "Created safe policy " << policy_num << endl;
            cout << "Theta_c: ";
            for(int j=0; j<theta_c.size(); j++){
                cout << theta_c[j] << "\t";
            }
            cout << endl;

            // write the policy params to a csv file
            std::ostringstream filename;
            filename << "/home/hao/CLionProjects/cs687_final_project/outputs/" << policy_num << ".csv";
            std::fstream fout;
            fout.open(filename.str().c_str(),std::fstream::out);
            for(int i=0; i<theta_c.size(); i++){
                fout << theta_c[i] << ",";
            }
            fout << endl;
            fout.close();

            policy_num ++;
            if(policy_num > num_of_trials){
                isDone = true; // we have produced enough safe policies
            }

        }
        else{
            cout << "Candidate Policy is not safe!" << endl;
        }
    }

}

