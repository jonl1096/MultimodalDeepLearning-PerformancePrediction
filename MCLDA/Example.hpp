#ifndef Example_hpp
#define Example_hpp

#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Example{
    public:
        bool cval;
        vector<string> words;
        int numTotalWords;
        
        Example(string);
        // ~Example();
        int getNumWords();
        string getString();
};

#endif /* Example_hpp */