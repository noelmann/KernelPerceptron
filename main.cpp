#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

#include "PerceptronClassifier.h"
#include "Sample.h"
#include "kernel_rbf.h"
#include "kernel_polynomial.h"
#include "kernel_linear.h"


using namespace std;

vector<Sample> trainingSet;

//splits the given string into multiple strings at the given character
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

//lodas the training data line by line and uses the last column as the label
void loadTrainingSet()
{

        string fullFilePath = R"(D:\QT-Projekte\KernelPerceptron\Dataset_NonLinear.txt)";

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

            Sample t(lineCount-1, features,label);
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
    std::cout << "Training started!" << std::endl;
    loadTrainingSet();
    kernel_linear linear = kernel_linear();
    kernel_polynomial polynomial = kernel_polynomial(2);
    kernel_rbf rbf = kernel_rbf(1.0);
    PerceptronClassifier c = PerceptronClassifier(trainingSet,rbf);
    cout << "Training completed" << endl;
    getchar();
    return 0;
}