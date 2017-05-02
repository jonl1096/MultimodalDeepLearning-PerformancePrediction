#include "Example.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <cstdio>

using namespace std;

Example::Example(string str) {
    istringstream iss(str);
    vector<string> tokens{istream_iterator<string>{iss},
                           istream_iterator<string>{}};
    this->cval = ((int) tokens[0][0]) - 48;
    vector<string>::const_iterator first = tokens.begin() + 1;
    vector<string>::const_iterator last = tokens.end();
    this->words = vector<string>(first, last);
    
    this->numTotalWords = this->words.size();
}

// Example::~Example(){
//     delete this->words;
// }

int Example::getNumWords() {
    return this->numTotalWords;
}

string Example::getString(){
    string str = "cval = ";
    str = str + to_string(this->cval);
    str = str + "\nwords = ";
    for(int i = 0; i < this->getNumWords(); i++){
        str = str + (this->words)[i] + ", ";
    }
    return str + "\n";
}