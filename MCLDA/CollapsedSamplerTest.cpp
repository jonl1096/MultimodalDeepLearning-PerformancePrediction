#include "CollapsedSampler.hpp"
#include "Example.hpp"
#include <string>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cstdio>

using std::endl;
using std::cout;
using std::istringstream;
using std::ostringstream;

// bool operator== (const FileDir &dir1, const FileDir &dir2)
// {
//     if (dir1.getName() == dir2.getName() && dir1.getSize() == dir2.getSize() && dir1.isFile() == dir2.isFile() )
//     {
//         return true;
//     }
//     return false;
// }

// Tests constructors and accessors:
//
// - A constructor which is passed a string name, a long size and a
// boolean value which is true if the item is a directory and false if
// it's a file. Include default parameter values of 4 for the size and
// false for boolean.
//
// - A copy constructor.
//
// - Functions that return the size and the name of the item, long
// getSize() and string getName() respectively.
//
// - A function called "isFile" that returns true if the object is a
// file, and false if it is a directory.

void exampleTest() {
    Example e("0 zero one two repeat repeat five six seven");
    assert(e.cval == 0);
    assert(e.words.at(0) == "zero");
    assert(e.words.at(1) != "two");
    // assert(e.numTotalWords == 8);
    // assert(e.numUniqueWords == 7);
    // assert(e.wordToIndex["two"] == 2);
    // assert(e.wordToIndex["five"] == 4);
    // assert(e.wordCounts.at(0) == 1);
    // assert(e.wordCounts.at(3) == 2);
    // assert(e.wordProbs.at(0) == 0.125);
    // assert(e.wordProbs.at(3) == 0.25);
    
    // Example e2("0 word word word word");
}

// TRAIN-FILE TEST-FILE OUTPUT-FILE K LAMBDA ALPHA BETA TOTAL-ITER BURN-IN-ITER
void constructorTest() {
    CollapsedSampler cs("train.txt", "test.txt", "output.txt", 5, 0.5, 0.1, 0.01, 1100, 1000);
    
    assert(cs.trainFileName == "train.txt");
    assert(cs.testFileName == "test.txt");
    assert(cs.K == 5);
}

void readInputTest() {
    CollapsedSampler cs("train.txt", "test.txt", "output.txt", 5, 0.5, 0.1, 0.01, 1100, 1000);
    cs.readInput();
    // checking if number of trainingExamples is right
    // passes even if there is a new line at the end of the file :D
    //cout << cs.trainingExamples.size() << endl;
    assert(cs.trainingExamples.size() == 3);
    // printf("Number of words i");
    // checking if number of words is right
    assert(cs.trainingExamples.at(0)->words.size() == 4);
    // checking if the first line is read correctly
    assert(cs.trainingExamples.at(0)->cval == 0);
    assert(cs.trainingExamples.at(0)->words.at(0) == "this");
    // checking if third line is read correctly
    assert(cs.trainingExamples.at(2)->cval == 1);
    assert(cs.trainingExamples.at(2)->words.at(0) == "heyooo");
    assert(cs.trainingExamples.at(2)->words.at(4) == "boi");
    assert(!(cs.trainingExamples.at(2)->words.at(2) == "ya"));
    
    assert(cs.testingExamples.size() == 3);
    // printf("Number of words i");
    // checking if number of words is right
    assert(cs.testingExamples.at(0)->words.size() == 2);
    // checking if the first line is read correctly
    assert(cs.testingExamples.at(0)->cval == 0);
    assert(cs.testingExamples.at(0)->words.at(0) == "hello");
    // checking if third line is read correctly
    assert(cs.testingExamples.at(2)->cval == 0);
    assert(cs.testingExamples.at(2)->words.at(0) == "what");
    assert(cs.testingExamples.at(2)->words.at(4) == "wazzup");
}

void initializeValuesTest() {
    CollapsedSampler cs("train.txt", "test.txt", "output.txt", 5, 0.5, 0.1, 0.01, 1100, 1000);
    cs.readInput();
    cs.initializeValues(cs.trainingExamples, cs.Ztrain, cs.Xtrain);
    cs.initializeValues(cs.testingExamples, cs.Ztest, cs.Xtest);
    assert(cs.Ztrain.size() == 3);
    assert(cs.Ztrain.at(0).size() == 4);
    assert(cs.Ztrain.at(2).size() == 5);
    assert(cs.Xtrain.size() == 3);
    assert(cs.Xtrain.at(0).size() == 4);
    assert(cs.Ztest.size() == 3);
    assert(cs.Ztest.at(0).size() == 2);
    assert(cs.Ztest.at(2).size() == 5);
    assert(cs.Xtest.size() == 3);
    assert(cs.Xtest.at(0).size() == 2);
    assert(cs.Ztrain[0][0] >= 0 && cs.Ztrain[0][0] <= 4);
    assert(cs.Xtest[0][0] >= 0 && cs.Xtest[0][0] <= 1);
    cout << cs.Ztrain[0][0] << " " << cs.Ztrain[0][1] << " " << cs.Xtrain[0][0] << " " << cs.Xtrain[0][1] << endl;
    //cout << cs.wordToIndex << endl;
    // Testing vocab stuff
    printf("%d\n", cs.wordToIndex.size());
    cout << cs.trainingExamples.at(2)->words.at(0) << endl;
    // printf("%s\n", cs.trainingExamples.at(2)->words.at(0));
    assert(cs.wordToIndex.size() == 16);
}
int main(void) {
    cout << "Running CollapsedSampler tests..." << endl;
    exampleTest();
    constructorTest();
    readInputTest();
    initializeValuesTest();
    //    writeTest();   // not required
    // toStringTest();
    cout << "CollapsedSampler tests passed." << endl;
    return 0;
}