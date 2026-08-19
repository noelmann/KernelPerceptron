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
#include "hyperparametergenerator.h"
#include <map>


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
void loadTrainingSet(const double &bias, const bool isLabelLast, const int maxNum)
{

        string fullFilePath = R"(D:\QT-Projekte\KernelPerceptron\mnist_train.csv)";

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
            double label;

            if(isLabelLast)
            {
                //Last number in line is treated as label
                label = stod(sample[sample.size()-1]);
                for (int i = 0;i<sample.size()-1;i++)
                {
                    features.push_back(stod(sample[i]));
                    //cout << stod(sample[i]) << endl;
                }
            }
            else
            {
                label = stod(sample[0]);
                for (int i = 1;i<sample.size();i++)
                {
                    features.push_back(stod(sample[i]));
                    //cout << stod(sample[i]) << endl;
                }
            }


            features.push_back(bias);



            Sample t(lineCount-1, features,label);
            trainingSet.push_back(t);
            /*for(int i = 0;i<t.getFeatureVector().size()-1;i++)
            {
                cout<<t.getFeatureVector()[i] << " ";

            }*/
            //cout << "Label:" << t.getLabel() << endl;

            if(lineCount==maxNum)
            {
                break;
            }
        }


        // Close the file
        MyReadFile.close();

}

void oneVSall()
{
    map<double, vector<Sample>> trainingSets;
    map<double, PerceptronClassifier> classifiers;

    // Find all classes
    for (const auto& s : trainingSet)
    {
        trainingSets[s.getLabel()];
    }

    // Create one binary dataset per class
    for (auto& pair : trainingSets)
    {
        double targetClass = pair.first;

        for (const auto& s : trainingSet)
        {
            double label = (s.getLabel() == targetClass) ? 1.0 : 0.0;

            pair.second.emplace_back(s.getIdentifier(),s.getFeatureVector(),label);
        }
    }

    for (const auto& pair : trainingSets)
    {
        cout << "Classifier for class " << pair.first
             << ": " << pair.second.size()
             << " samples" << endl;

        kernel_linear k = kernel_linear();
        PerceptronClassifier c = PerceptronClassifier(k);
        //c.hyperParameterGridSearch(pair.second,pair.second,100,1,10,1);
        c.train(pair.second,5000);
        classifiers.emplace(pair.first, c);
    }


    double classification;
    double maximumDistance = -1000;
    double t;

    for(int i =0;i<trainingSet.size();i++)
    {
        double maximumDistance = -1000;
        for(auto &pair : classifiers)
        {
            t=pair.second.classify(trainingSet[i],false);
            if(t > maximumDistance)
            {
                maximumDistance = t;
                classification=pair.first;
            }
        }

        cout << "Classification result:" << classification << "| True label:" << trainingSet[i].getLabel() << endl;

    }





}

int main()
{
    /*vector<Entry> c = HyperparameterGenerator::generateHyperparameterSet(3,-3,3,0.5);

    for(const auto &e: c)
    {
        string entry="";
        for(const double &d :e.values)
        {
            entry += to_string(d) + " ";
        }

        cout << "|" << entry <<"|" << endl;
    }

    cout << "Totalcount:" << c.size() << endl;*/

    cout << "Loading training set" << endl;
    loadTrainingSet(1, false,500);
    oneVSall();
    getchar();
    return 0;
}