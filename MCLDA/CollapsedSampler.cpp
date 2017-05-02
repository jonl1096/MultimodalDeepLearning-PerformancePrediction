#include "CollapsedSampler.hpp"
#include "Example.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#include <unordered_map>

using namespace std;

CollapsedSampler::CollapsedSampler(string tr, string te, string ou, int k,
    float l, float a, float b, int totIter, int burnIter) {
    this->trainFileName = tr;
    this->testFileName = te;
    this->outputFileName = ou;
    this->K = k;
    this->lambda = l;
    this->alpha = a;
    this->beta = b;
    this->numTotalIter = totIter;
    this->numBurnInIter = burnIter;
    
    this->wordToIndex = unordered_map<string, int>();
    this->C = 0;
    
    this->trainingExamples = vector<Example*>();
    this->testingExamples = vector<Example*>();
    
    this->Ztrain = vector<vector<int> >();
    this->Xtrain = vector<vector<int> >();
    this->Ztest = vector<vector<int> >();
    this->Xtest = vector<vector<int> >();
}

// CollapsedSampler::~CollapsedSampler() {
// }

void CollapsedSampler::readInput() {
    string line;
    ifstream myTrainfile(this->trainFileName);
    // vector of trainingExamples to be returned
    // vector<Example*>* trainingExamples = new vector<Example*>();
    // myfile being a collection of trainingExamples
    if (myTrainfile.is_open()) {
        // each line is an example
        while (getline(myTrainfile, line)) {
            Example* tempExample = new Example(line);
            this->trainingExamples.push_back(tempExample);
        }
        myTrainfile.close();
    } else cout << "Unable to open file";
    
    ifstream myTestfile(this->testFileName);
    // vector of trainingExamples to be returned
    // vector<Example*>* trainingExamples = new vector<Example*>();
    // myfile being a collection of trainingExamples
    if (myTestfile.is_open()) {
        // each line is an example
        while (getline(myTestfile, line)) {
            Example* tempExample = new Example(line);
            this->testingExamples.push_back(tempExample);
        }
        myTestfile.close();
    } else cout << "Unable to open file";
    
    // this->trainingExamples = trainingExamples;
}

/**
 * Function used to initialize the Z and X 2d vectors.
 * This is step 1 according to the algorithm given on page 6.
 * Need to call this twice, once for train and once for test.
 */
void CollapsedSampler::initializeValues(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X){
    // numExamples is number of rows
    int numExamples = examples.size();
    // numWords is number of columns for that row
    int numWords;
    int wordIndex = this->wordToIndex.size();
    // Initializing Z and X
    srand(time(NULL));  // initialize random seed
    for (int i = 0; i < numExamples; i++) {
        int c = examples.at(i)->cval;
        if (c >= this->C){
            this->C = c+1;
        }
        numWords = examples.at(i)->getNumWords();
        vector<int> zRow = vector<int>();
        vector<int> xRow = vector<int>();
        // Initializing z_{d,i} to a random integer in [0, k-1] and
        // x_{d,i} to a random interger in [0, 1]
        for (int j = 0; j < numWords; j++) {
            string word = examples[i]->words[j];
            if(!this->wordToIndex.count(word)){
                this->wordToIndex[word] = wordIndex;
                wordIndex++;
            }
            // generate random number in [0, k-1], and adding it to zRow:
            int zRand = rand() % this->K;
            // printf("xRand: %d, ", zRand);
            zRow.push_back(zRand);
            // generate random number between in [0, 1], and adding it to zRow:
            int xRand = rand() % 2;
            // printf("xRand: %d\n", xRand);
            xRow.push_back(xRand);
        }
        // adding the rows
        Z.push_back(zRow);
        X.push_back(xRow);
    }
    // return 0;
}

void CollapsedSampler::initializeVariables(){
    this->thetaTrain = vector<vector<float> >(this->trainingExamples.size());
    this->thetaTest = vector<vector<float> >(this->testingExamples.size());
    this->phi = vector<vector<float> >(this->K);
    this->phic = vector<vector<vector<float> > >(this->C);
    this->thetaTrain_avg = vector<vector<float> >(this->trainingExamples.size());
    this->thetaTest_avg = vector<vector<float> >(this->testingExamples.size());
    this->phi_avg = vector<vector<float> >(this->K);
    this->phic_avg = vector<vector<vector<float> > >(this->C);
    
    for(int d = 0; d < this->trainingExamples.size(); d++){
        this->thetaTrain[d] = vector<float>(this->K);
        this->thetaTrain_avg[d] = vector<float>(this->K);
    }
    for(int d = 0; d < this->testingExamples.size(); d++){
        this->thetaTest[d] = vector<float>(this->K);
        this->thetaTest_avg[d] = vector<float>(this->K);
    }
    for(int k = 0; k < this->K; k++){
        this->phi[k] = vector<float>(this->wordToIndex.size());
        this->phi_avg[k] = vector<float>(this->wordToIndex.size());
    }
    for(int c = 0; c < this->C; c++){
        this->phic[c] = vector<vector<float> >(this->K);
        this->phic_avg[c] = vector<vector<float> >(this->K);
        for(int k = 0; k < this->K; k++){
            this->phic[c][k] = vector<float>(this->wordToIndex.size());
            this->phic_avg[c][k] = vector<float>(this->wordToIndex.size());
        }
    }
}

void CollapsedSampler::initializeCounts(){
    //set counts to 0
    // cout << "set counts to 0" << endl;
    this->ndk_train = vector<vector<int> >(this->trainingExamples.size());
    this->nd_train = vector<int>(this->trainingExamples.size(), 0);
    for(int d = 0; d < this->trainingExamples.size(); d++){
        this->ndk_train[d] = vector<int>(this->K, 0);
    }
    
    this->ndk_test = vector<vector<int> >(this->testingExamples.size());
    this->nd_test = vector<int>(this->testingExamples.size(), 0);
    for(int d = 0; d < this->testingExamples.size(); d++){
        this->ndk_test[d] = vector<int>(this->K, 0);
    }
    
    this->nkw_train = vector<vector<int> >(this->K);
    this->nk_train =  vector<int>(this->K, 0);
    for(int k = 0; k < this->K; k++){
        this->nkw_train[k] = vector<int>(this->wordToIndex.size(), 0);
    }
    
    this->nckw_train = vector<vector<vector<int> > >(this->C);
    this->nck_train = vector<vector<int> >(this->C);
    for(int c = 0; c < this->C; c++){
        this->nckw_train[c] = vector<vector<int> >(this->K);
        this->nck_train[c] = vector<int>(this->K, 0);
        for(int k = 0; k < this->K; k++){
            this->nckw_train[c][k] = vector<int>(this->wordToIndex.size(), 0);
        }
    }
    
    //count counts
    // cout << "count counts" << endl;
    //Train Counts
    // cout << "counting training counts" << endl;
    for (int d = 0; d < this->trainingExamples.size(); d++) {
        int numWords = this->trainingExamples.at(d)->getNumWords();
        for(int i = 0; i < numWords; i++){
            this->updateCounts(this->trainingExamples, this->Ztrain, this->Xtrain, d, i, true, true);
        }
    }
    
    //Test Counts
    // cout << "counting testing counts" << endl;
    for (int d = 0; d < this->testingExamples.size(); d++) {
        // cout << "document" << endl;
        int numWords = this->testingExamples.at(d)->getNumWords();
        for(int i = 0; i < numWords; i++){
            // cout << "document word" << endl;
            //cout << this->Ztest.size() << this->Ztest[d].size() << endl;
            this->updateCounts(this->testingExamples, this->Ztest, this->Xtest, d, i, false, true);
        }
    }
}

/**
 * Compute Log-likelihood of the corpus.
 */
float CollapsedSampler::logLikelihood(float lambda, 
                    vector<Example*>& examples,
                    vector<vector<float> >& th,
                    vector<vector<float> >& p,
                    vector<vector<vector<float> > >& pc){
    
    float likelihood = 0;
    //for each document
    for(int d = 0; d < th.size(); d++){
        int c = examples[d]->cval;
        int numWords = examples[d]->getNumWords();
        //for each word in a document
        for(int i = 0; i < numWords; i++){
            //for each topic
            int w = this->wordToIndex[examples[d]->words[i]];
            float prob = 0.0;
            for(int k = 0; k < this->K; k++){
                //get cur theta
                
                //cout << "FEAR" << endl;
                float curtheta = th[d][k];
                //cout << "there" << endl;
                float curphi = p[k][w];
                //cout << "here" << endl;
                float curphic = pc[c][k][w];
                //cout << "heythere" << endl;
                
                prob += (curtheta*((1-lambda)*curphi + lambda*curphic));
                //cout << "lkhood: " << likelihood << endl;
                
            }
            //cout << "sun" << endl;
            likelihood += log(prob);
            
            //cout << "none" << endl;
        }
    }
    return likelihood;
}
/**
 * Calculate full conditional for x_di.
 * lambda   lambda
 * n1       n^z_w
 * n2       n^z_*
 * V        vocab size
 * b        beta
 * 
 */
float CollapsedSampler::fullxConditional(float lambda, int n1, int n2, int V, float b){
    
    return lambda*this->map(b,V,n1,n2);
}

/**
 * Calculate full conditional for z_di.
 * a    alpha
 * b    beta
 * K    number of topics
 * V    vocab size
 * n1   n^d_k
 * n2   n^d_*
 * n3   n^k_w or n^ck_w
 * n4   n^k_* or n^ck_*
 */
float CollapsedSampler::fullzConditional(float a, float b, int K, int V, int n1, int n2, int n3, int n4){
    
    return this->map(a,K,n1,n2) * this->map(b,V,n3,n4);
}
/**
 * Compute MAP estimation
 */ 
float CollapsedSampler::map(float a, int K, int n1, int n2){
    return (n1 + a)/(n2 + (K * a));
}

/**
 * 
 * 
 */
void CollapsedSampler::updateCounts(vector<Example*>& examples, vector<vector<int> >& Z, 
    vector<vector<int> >& X, int d, int i, bool trainOrTest, bool incrementDecrement){
    int updater = 0;
    if (incrementDecrement){
        updater = 1;
    } else {
        updater = -1;
    }
    //cout << "getting k" << endl;
    int k = Z[d][i];
    // cout << "getting c" << endl;
    int c = examples[d]->cval;
    //counts for train
    if (trainOrTest){
        //cout << "updating traning counts" << endl;
        //cout << "getting word index" << endl;
        int w = this->wordToIndex[examples[d]->words[i]];
        //cout << "1" << endl;
        this->ndk_train[d][k]+=updater;
        //cout << "2" << endl;
        this->nd_train[d]+=updater;
        //cout << "3" << endl;
        this->nkw_train[k][w]+=updater;
        //cout << "4" << endl;
        this->nk_train[k]+=updater;
        //cout << "5" << endl;
        this->nckw_train[c][k][w]+=updater;
        //cout << "6" << endl;
        this->nck_train[c][k]+=updater;
    }
    //counts for test;
    else {
        //cout << "updating testing counts" << endl;
        this->ndk_test[d][k]+=updater;
        // cout << "1" << endl;
        this->nd_test[d]+=updater;
        // cout << "2" << endl;
    }
    //cout << "done updating counts" << endl;
}

int CollapsedSampler::randomlySample(vector<int>& values, vector<float>& probabilities){
    //Normalize probabilities
    //May not need this.  Mostly for debuging.
    float sum = 0;
    for(int i = 0; i < probabilities.size(); i++){
        sum += probabilities[i];
    }
    for(int i = 0; i < probabilities.size(); i++){
        probabilities[i] /= sum;
    }
    
    //get random value
    sum = 0;
    //srand(time(NULL));  // initialize random seed
    float num = (float)rand() / (float)RAND_MAX;
    //cout << num << endl;
    for(int i = 0; i < probabilities.size(); i++){
        sum += probabilities[i];
        //cout << sum << endl;
        if(num <= sum){
            //cout << values[i] << endl;
            return values[i];
        }
    }
    return values[probabilities.size()-1];
}

/**
 * examples     documents
 * Z            z values
 * X            x values
 * d            current document index
 * i            current word
 */
void CollapsedSampler::updateValues(vector<Example*>& examples, vector<vector<int> >& Z, 
    vector<vector<int> >& X, int d, int i, bool trainOrTest){
    //Set Counts
    vector<vector<int> >* ndk;
    vector<int>* nd;
    if (trainOrTest){
        ndk = &this->ndk_train;
        nd = &this->nd_train;
    } else {
        ndk = &this->ndk_test;
        nd = &this->nd_test;
    }
    
    //set c and w
    int c = examples[d]->cval;
    int w = this->wordToIndex[examples[d]->words[i]];
    
    
    //Sample Z
    vector<float> probs = vector<float>(this->K);
    vector<int> values = vector<int>(this->K);
    if(X[d][i] == 0){
        for(int k = 0; k < this->K; k++){
            probs[k] = this->fullzConditional(this->alpha, this->beta, this->K, this->wordToIndex.size(), 
                ndk->at(d)[k], nd->at(d), this->nkw_train[k][w], this->nk_train[k]);
            values[k] = k;
        }
    } else {
        for(int k = 0; k < this->K; k++){
            probs[k] = this->fullzConditional(this->alpha, this->beta, this->K, this->wordToIndex.size(), 
                ndk->at(d)[k], nd->at(d), this->nckw_train[c][k][w], this->nck_train[c][k]);
            values[k] = k;
        }
    }
    Z[d][i] = this->randomlySample(values, probs);
    
    //Sample X
    values = vector<int>(2);
    values[0] = 0; values[1] = 1;
    probs = vector<float>(2);
    int k = Z[d][i];
    probs[0] = this->fullxConditional(1-this->lambda, this->nkw_train[k][w], this->nk_train[k], this->wordToIndex.size(), this->beta);
    probs[1] = this->fullxConditional(this->lambda, this->nckw_train[c][k][w], this->nck_train[c][k], this->wordToIndex.size(), this->beta);
    X[d][i] = this->randomlySample(values, probs);
}

/**
 * Doing steps 2.a and 2.d.
 * 
 * @param examples
 */
void CollapsedSampler::doSampling(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X, bool trainOrTest){
    //for each document
    for(int d = 0; d < examples.size(); d++){
        int numWords = examples.at(d)->getNumWords();
        //iterating through each word in a document
        for(int i = 0; i < numWords; i++){
            //Exclude counts
            //  cout << "excluding counts" << endl;
            this->updateCounts(examples, Z, X, d, i, trainOrTest, false);
            
            //Update z_{d,i} and x_{d,i}
            //  cout << "updating values" << endl;
            this->updateValues(examples, Z, X, d, i, trainOrTest);
            
            //Include counts
            //  cout << "including counts" << endl;
            this->updateCounts(examples, Z, X, d, i, trainOrTest, true);
        }
    }
}

void CollapsedSampler::MAPEstimation(bool trainOrTest){
    if(trainOrTest){
        //MAP Estimate thetaTrain_dk
        for(int d = 0; d < this->trainingExamples.size(); d++){
            for(int k = 0; k < this->K; k++){
                this->thetaTrain[d][k] = map(this->alpha, this->K, this->ndk_train[d][k], this->nd_train[d]);
                //cout << "thetaTrain: " << this->thetaTrain[d][k] << endl;
            }
        }
        
        //MAP Estimate phi_kw
        for(int k = 0; k < this->K; k++){
            for(int w = 0; w < this->wordToIndex.size(); w++){
                this->phi[k][w] = map(this->beta, this->wordToIndex.size(), this->nkw_train[k][w], this->nk_train[k]);
                //cout << "phi: " << this->phi[k][w] << endl;
            }
        }
        
        
        //MAP Estimate phic_kw
        for(int c = 0; c < this->C; c++){
            for(int k = 0; k < this->K; k++){
                for(int w = 0; w < this->wordToIndex.size(); w++){
                    this->phic[c][k][w] = map(this->beta, this->wordToIndex.size(), this->nckw_train[c][k][w], this->nck_train[c][k]);
                    //cout << "phic: " << this->phic[c][k][w] << endl;
                }
            }
        }
    } else {
        //MAP Estimate thetaTest_dk
        for(int d = 0; d < this->testingExamples.size(); d++){
            for(int k = 0; k < this->K; k++){
                this->thetaTest[d][k] = map(this->alpha, this->K, this->ndk_test[d][k], this->nd_test[d]);
            }
        }
    }
}

void CollapsedSampler::updateAverageValues(int t){
    //compute average thetaTrain_dk
    for(int d = 0; d < this->trainingExamples.size(); d++){
        for(int k = 0; k < this->K; k++){
            this->thetaTrain_avg[d][k] = (this->thetaTrain_avg[d][k]*(t-this->numBurnInIter-1) + this->thetaTrain[d][k])/(t-this->numBurnInIter);
        }
    }
    
    //compute average thetaTest_dk
    for(int d = 0; d < this->testingExamples.size(); d++){
        for(int k = 0; k < this->K; k++){
            this->thetaTest_avg[d][k] = (this->thetaTest_avg[d][k]*(t-this->numBurnInIter-1) + this->thetaTest[d][k])/(t-this->numBurnInIter);
        }
    }
    
    //compute average phi_kw
    for(int k = 0; k < this->K; k++){
        for(int w = 0; w < this->wordToIndex.size(); w++){
            this->phi_avg[k][w] = (this->phi_avg[k][w]*(t-this->numBurnInIter-1) + this->phi[k][w])/(t-this->numBurnInIter);
        }
    }
    
    //compute average phi_ckw
    for(int c = 0; c < this->C; c++){
        for(int k = 0; k < this->K; k++){
            for(int w = 0; w < this->wordToIndex.size(); w++){
                this->phic_avg[c][k][w] = (this->phic_avg[c][k][w]*(t-this->numBurnInIter-1) + this->phic[c][k][w])/(t-this->numBurnInIter);
            }
        }
    }
}
/**
 * Makes the 6 output files needed.
 * 
 */
void CollapsedSampler::makeOutput(){
    
    ofstream thetaf(this->outputFileName + "-theta");
    ofstream phif(this->outputFileName + "-phi");
    ofstream phi0f(this->outputFileName + "-phi0");
    ofstream phi1f(this->outputFileName + "-phi1");

    
    thetaf.precision(13);
    phif.precision(13);
    phi0f.precision(13);
    phi1f.precision(13);
    
    //Write Thetafile
    //for each document
    for(int i = 0; i < this->thetaTrain_avg.size(); i++){
        //for each word in the document
        for(int j = 0; j < this->thetaTrain_avg[i].size(); j++){
            
            thetaf << scientific << this->thetaTrain_avg[i][j] << " ";
        }
        thetaf << endl;
    }
    
    //Write PhiFiles
    //for each word
    unordered_map<string,int>::iterator it;
    
    for(it = wordToIndex.begin();it != wordToIndex.end();it++){
        //for each topic
        phif << it->first << " ";
        phi0f << it->first << " ";
        phi1f << it->first << " ";
        
        int currindex = it->second;
        for(int b = 0; b < this->phi_avg.size();b++){
            
            phif << scientific << this->phi_avg[b][currindex] << " ";
        }
        phif << endl;
    
        //for each topic in phi C
        for(int c = 0; c < this->phic_avg[0].size();c++){
            
            phi0f << scientific << this->phic_avg[0][c][currindex] << " ";
            phi1f << scientific << this->phic_avg[1][c][currindex] << " ";
        }
        phi0f << endl;
        phi1f << endl;
        
    }
    
    thetaf.close();
    phif.close();
    phi0f.close();
    phi1f.close();
    
    
}
/**
 * This is step 2 according to the algorithm given on page 6.
 */
 /**
  * For each itr t:
  *     (a)For each token (d,i) in each document d in training set:
  *         i. Update counts to exclude assignments of the current token
  *         ii. Randomly sample new value for z_di = k. 
  *         iii. Randomly sample new value for x_di. Use z_di
  *         iv. Update counts to enclude new sampled assignments of current token
  *     (b)Estimate parameters
  *     (c)If burn in passed:
  *         Incorporate the estimated parameters into the estimate of the exptected val
  *     (d)Sample Z and X of the test, but use Phi_hat from curr itr rather than recount
  *     (e)Compute the train log-likelihood
  *     (f)Compute the test log-likelihood
  */
void CollapsedSampler::doIterations() {
    
    //Make outputfiles
    ofstream trainf(this->outputFileName + "-trainll");
    ofstream testf(this->outputFileName + "-testll");
    trainf.precision(13);
    testf.precision(13);
    
    //Iterations
    for(int t = 0; t < this->numTotalIter; t++){
        //For Training Documents
        //cout << "training sampling" << endl;
        this->doSampling(this->trainingExamples,this->Ztrain,this->Xtrain,true);
        
        //Map Estimation
        //cout << "MAP Estimation" << endl;
        this->MAPEstimation(true);
        
        //Expected Value calculation
        //cout << "Expected Value Calculation" << endl;
        if(t > this->numBurnInIter){
            this->updateAverageValues(t);
        }
        
        //Update for test set
        //cout << "testing sampling" << endl;
        this->doSampling(this->testingExamples,this->Ztest,this->Xtest,false);
        //cout << "ESTIMATING" << endl;
        this->MAPEstimation(false);
        //cout << "DONE ESTIMATING" << endl;
        //compute log-likelihoods
        //cout << "Computing training Log-likelihood" << endl;
        //cout << "train logLikelihood: " << this->logLikelihood(this->lambda, this->trainingExamples, this->thetaTrain, this->phi, this->phic) << endl;
        trainf << scientific << this->logLikelihood(this->lambda, this->trainingExamples, this->thetaTrain, this->phi, this->phic) << endl;
        //cout << "Computing testing Log-likelihood" << endl;
        //cout << "test logLikelihood: " << this->logLikelihood(this->lambda, this->testingExamples, this->thetaTest, this->phi, this->phic) << endl;
        testf << scientific << this->logLikelihood(this->lambda, this->testingExamples, this->thetaTest, this->phi, this->phic) << endl;
        //if((t+1) % 10 == 0){
        cout << "end of iteration " << (t+1) << endl;
        //cout << "ndk00 " << this->ndk_train[1][0] << endl;
        //cout << "nd0 " << this->nd_train[1] << endl;
    }
}