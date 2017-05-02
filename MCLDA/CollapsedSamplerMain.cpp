#include "CollapsedSampler.hpp"
#include "Example.hpp"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    // TRAIN-FILE TEST-FILE OUTPUT-FILE K LAMBDA ALPHA BETA TOTAL-ITER BURN-IN-ITER
    //   2          3           4       5   6       7   8       9           10
    if (argc != 10){
        cout << "NOT ENOUGH COMMAND LINE ARGS" << endl;
        cout << "EXITING..." << endl;
    } else {
        cout << "RUNNING OUR GNARLY GIBBS SAMPLER~!" << endl;
        //CollapsedSampler cs = CollapsedSampler("../hw3b-files/intput-train.txt", "../hw3b-files/intput-test.txt", "output.txt", 10, 0.5, 0.1, 0.01, 1100, 1000);
        
        string inputTrain = (string) argv[1];
        string inputTest = (string) argv[2];
        string outputFile = (string) argv[3];
        int K = stoi(argv[4]);
        float lambda = stof(argv[5]);
        float alpha = stof(argv[6]);
        float beta = stof(argv[7]);
        int totalIter = stoi(argv[8]);
        int burnInIter = stoi(argv[9]);
        CollapsedSampler cs = CollapsedSampler(inputTrain, inputTest, outputFile, K, lambda, alpha, beta, totalIter, burnInIter);
        
        cout << "reading input" << endl;
        cs.readInput();
        cout << "initializing train" << endl;
        cs.initializeValues(cs.trainingExamples, cs.Ztrain, cs.Xtrain);
        cout << "initializing test" << endl;
        cs.initializeValues(cs.testingExamples, cs.Ztest, cs.Xtest);
        cout << "intitializing variables" << endl;
        cs.initializeVariables();
        cout << "initializing counts" << endl;
        cs.initializeCounts();
        cs.MAPEstimation(true);
        cout << "train logLikelihood: " << cs.logLikelihood(cs.lambda, cs.trainingExamples, cs.thetaTrain, cs.phi, cs.phic) << endl;
        cout << "doing Iterations" << endl;
        cs.doIterations();
        cout << "done" << endl;
        cout << "writing output" << endl;
        cs.makeOutput();
        cout << "done" << endl;
        cout << "converged train logLikelihood: " << cs.logLikelihood(cs.lambda, cs.trainingExamples, cs.thetaTrain_avg, cs.phi_avg, cs.phic_avg) << endl;
        cout << "converged test logLikelihood: " << cs.logLikelihood(cs.lambda, cs.testingExamples, cs.thetaTest_avg, cs.phi_avg, cs.phic_avg) << endl;
    }
    return 0;
}

// int main(int argc, char *argv[]){
    // cout << "hello" << endl;
    // vector<Example*>* vect = readInput("test.txt");
    // cout << "hello" << 7 << endl;
    // cout << (*vect)[0]->getString()<< endl;
    // cout << "hello" << 7 << endl;
    // vector<Example*>* examples = readInput("../hw3b-files/input-train.txt");
    // vector<vector<int> >* Z = new vector<vector<int> >(examples->size());
    // vector<vector<int> >* X = new vector<vector<int> >(examples->size());
    // initializeValues(Z, X, examples, 10);
    // cout << Z->at(0)[0] << endl;
    
    
    // TRAIN-FILE TEST-FILE OUTPUT-FILE K LAMBDA ALPHA BETA TOTAL-ITER BURN-IN-ITER
    //   2          3           4       5   6       7   8       9           10
    // if (argc != 10){
    //     cout << "NOT ENOUGH COMMAND LINE ARGS" << endl;
    //     cout << "EXITING..." << endl;
    // } else {
    //     cout << "RUNNING OUR GNARLY GIBBS SAMPLER~!" << endl;
    //     //runGibsSampling(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);
    // }
//     return 0;
// }