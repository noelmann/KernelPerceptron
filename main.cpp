#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

#include "PerceptronClassifier.h"
#include "Sample.h"
#include "kernel_rbf.h"
#include "kernel_polynomial.h"
#include "kernel_sigmoid.h"
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
void loadTrainingSet(const double &bias)
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

            features.push_back(bias);

            //Last number in line is treated as label
            double label = stod(sample[sample.size()-1]);

            Sample t(lineCount-1, features,label);
            trainingSet.push_back(t);
            for(int i = 0;i<t.getFeatureVector().size()-1;i++)
            {
                cout<<t.getFeatureVector()[i] << " ";

            }
            cout << "Label:" << t.getLabel() << endl;

        }


        // Close the file
        MyReadFile.close();

}

double calculateAccuracy(PerceptronClassifier &classifier, const vector<Sample> &testset)
{
    double correctCounter = 0;
    for(int i =0;i<testset.size();i++)
    {
        if(classifier.classify(testset[i]) == testset[i].getLabel())
        {
            //cout << classifier.classify(testset[i]) << endl;
            correctCounter++;
        }
    }

    return correctCounter/testset.size();
}

void tuneSigmoidKernel(double (&sigmoidParameters)[2], const vector<Sample> tuningSet)
{
    double bestAcc = 0.0;
    double temp;
    double stepSize = 0.01;
    double a = sigmoidParameters[0];
    double c = sigmoidParameters[1];
    for(int x=1;x<100;x++)
    {
        for(int y=1;y<100;y++)
        {
            PerceptronClassifier classifier = PerceptronClassifier(tuningSet,100,kernel_sigmoid(a,c));
            temp = calculateAccuracy(classifier,tuningSet);
            if(temp > bestAcc)
            {
                bestAcc = temp;
                sigmoidParameters[0] =a;
                sigmoidParameters[1] =c;
            }

            a=x*stepSize;
            c=y*stepSize;
        }
    }
}


int main() {
    std::cout << "Training started!" << std::endl;
    loadTrainingSet(0);
    kernel_linear linear = kernel_linear();
    kernel_polynomial polynomial = kernel_polynomial(2);
    kernel_rbf rbf = kernel_rbf(1);

    /*cout << "Training completed" << endl;
    cout << "Stopped training after " << c.getIterations() << " iterations." << std::endl;
    double acc = calculateAccuracy(c,trainingSet)*100;
    cout << "Accuracy on training set: " << acc << "%" << endl;*/


    cout << "Running hyperparameter tuning for sigmoid kernel." << endl;
    double params[2] = {0.01,0.01};
    tuneSigmoidKernel(params, trainingSet);
    cout << "Best slope: " << params[0] << endl;
    cout << "Best intercept: " << params[1] << endl;

    kernel_sigmoid sigmoid = kernel_sigmoid(params[0],params[1]);
    PerceptronClassifier c = PerceptronClassifier(trainingSet,100,sigmoid);
    double acc = calculateAccuracy(c,trainingSet);
    cout << "Accuracy on training set: " << acc << "%" << endl;
    getchar();
    return 0;
}