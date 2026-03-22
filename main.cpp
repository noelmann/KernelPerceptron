#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

#include "PerceptronClassifier.h"
#include "Sample.h"


using namespace std;

vector<Sample> trainingSet;


vector<string> split(const string &s, char delimiter)
{
    vector<string> substrings;
    string temp;
    //cout << s << endl;
    for (int i = 0;i<s.size()+1;i++)
    {
        if (i == s.size() || s[i] == delimiter)
        {
            substrings.push_back(temp);
            //cout << temp << endl;
            temp = "";
        }
        else if (s[i] != delimiter)
        {
            temp+=s[i];
        }
    }
    return substrings;
}


void loadTrainingSet()
{

        string fullFilePath = "C:\\Users\\DerWiggler01\\CLionProjects\\KernelPerceptron\\Dataset_Nonlinear.txt";

        cout << fullFilePath << endl;
        ifstream MyReadFile(fullFilePath);
        string text;
        int lineCount = 0;
        // Use a while loop together with the getline() function to read the file line by line
        while (getline (MyReadFile, text))
        {
            lineCount++;
            if (lineCount == 1)
            {
                //skips header line
                continue;
            }
            // Output the text from the file
            vector<string> sample = split(text,',');
            vector<double> features;

            for (int i = 0;i<sample.size()-1;i++)
            {
                features.push_back(stod(sample[i]));
                //cout << stod(sample[i]) << endl;
            }

            //Last number in line is treated as label
            double label = stod(sample[sample.size()-1]);

            Sample t(features,label);
            trainingSet.push_back(t);
            for(int i = 0;i<t.getFeatureVector().size();i++)
            {
                cout<<t.getFeatureVector()[i] << " ";

            }
            cout << "Label:" << t.getLabel() << endl;

        }

        // Close the file
        MyReadFile.close();

}


int main() {
    std::cout << "Hello, World!" << std::endl;
    loadTrainingSet();
    PerceptronClassifier c = PerceptronClassifier(trainingSet,true);
    cout << "Training completed" << endl;
    getchar();
    return 0;
}