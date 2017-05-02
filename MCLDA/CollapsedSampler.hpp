#ifndef CollapsedSampler_hpp
#define CollapsedSampler_hpp

#include "Example.hpp"
#include <vector>
#include <string>

using namespace std;

class CollapsedSampler {
    private:

    public:
        // Given values:
        string trainFileName;
        string testFileName;
        string outputFileName;
        int K;
        // not sure what the following three types should be so I just made them floats
        float lambda;
        float alpha;
        float beta;
        int numTotalIter;
        int numBurnInIter;
        
        // Calculated Values:
        unordered_map<string, int> wordToIndex;
        int C;

        
        vector<vector<float> > thetaTrain;   //document<topic<val>>
        vector<vector<float> > thetaTest;   //document<topic<val>>
        vector<vector<float> > phi;     //topic<word<val>>
        vector<vector<vector<float> > > phic;    //collection<topic<word<val>>>
        vector<vector<float> > thetaTrain_avg;   //document<topic<val>>
        vector<vector<float> > thetaTest_avg;   //document<topic<val>>
        vector<vector<float> > phi_avg;     //topic<word<val>>
        vector<vector<vector<float> > > phic_avg;    //collection<topic<word<val>>>
        vector<Example*> trainingExamples;
        vector<Example*> testingExamples;
        vector<vector<int> > Ztrain;  // both Z and X consist only of ints
        vector<vector<int> > Xtrain;
        vector<vector<int> > Ztest;
        vector<vector<int> > Xtest;
        //counts for train
        vector<vector<int> > ndk_train;
        vector<int> nd_train;
        vector<vector<int> > nkw_train;
        vector<int> nk_train;
        vector<vector<vector<int> > > nckw_train;
        vector<vector<int> > nck_train;
        //counts for test;
        vector<vector<int> > ndk_test;
        vector<int> nd_test;
        
        
        // Functions:
        CollapsedSampler(string tr, string te, string ou, int k,
            float l, float a, float b, int totIter, int burnIter);
        // ~CollapsedSampler();
        void readInput();
        void initializeVariables();
        void initializeValues(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X);
        void initializeCounts();
        float logLikelihood(float lambda, vector<Example*>& examples, vector<vector<float> >& th, vector<vector<float> >& p, vector<vector<vector<float> > >& pc);
        float fullxConditional(float lambda, int n1, int n2, int V, float b);
        float fullzConditional(float a, float b, int K, int V, int n1, int n2, int n3, int n4);
        float map(float a, int K, int n1, int n2);
        void updateCounts(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X, int d, int i, bool trainOrTest, bool incrementDecrement);
        int randomlySample(vector<int>& values, vector<float>& probabilities);
        void updateValues(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X, int d, int i, bool trainOrTest);
        void doSampling(vector<Example*>& examples, vector<vector<int> >& Z, vector<vector<int> >& X, bool trainOrTest);
        void MAPEstimation(bool trainOrTest);
        void updateAverageValues(int t);
        void doIterations();
        void makeOutput();
        // Old shit
        // int getNumWords();
        // string getString();
};

#endif /* CollapsedSampler_hpp */