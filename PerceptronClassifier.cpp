//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"
#include <cmath>
#include "hyperparametergenerator.h"

PerceptronClassifier::PerceptronClassifier(kernel &kernel) : k(kernel)
{
    partialError.clear();
    iterations = 0;
}

void PerceptronClassifier::train(const std::vector<Sample> &trainingSamples, const int &max_iterations)
{
    partialError.clear();
    iterations = 0;
    for (const auto& trainingSample : trainingSamples)
    {
        partialError.push_back({trainingSample, 0.0});
    }

    while (!checkIfClassifierIsPerfect() && iterations < max_iterations)
    {
        for (const auto& trainingSample : trainingSamples)
        {
            double error = (trainingSample.getLabel()-classify(trainingSample));
            incrementPartialError(trainingSample,error);
        }

        iterations++;
    }
}

void PerceptronClassifier::hyperParameterGridSearch(const std::vector<Sample> &trainingSamples,const std::vector<Sample> &developmentSamples, const int &max_iterations, double lowerBound, double upperBound, double stepSize)
{
    double bestAccuracy = -0.1;
    double acc = bestAccuracy;
    std::vector<double> bestVec;
    std::vector<double> vec;
    std::vector<Entry> hyperParameterSet = HyperparameterGenerator::generateHyperparameterSet(k.getParameterVector().size(),lowerBound,upperBound,stepSize);
    struct Entry standard;
    standard.values = k.getParameterVector();
    hyperParameterSet.insert(hyperParameterSet.begin(),standard);

    for(const struct Entry &e : hyperParameterSet)
    {
        vec.clear();

        for(const double &value : e.values)
        {
            vec.emplace_back(value);
        }

        k.setParameterVector(vec);

        train(trainingSamples,max_iterations);
        acc = calculateAccuracy(developmentSamples);

        if(acc > bestAccuracy)
        {
            bestAccuracy = acc;
            bestVec = vec;
        }
    }
    k.setParameterVector(bestVec);
    train(trainingSamples,max_iterations);
}


//classifies a given sample using a dual form perceptron in combination with a given kernel(linear/polynomial/rbf)
double PerceptronClassifier::classify(const Sample &t)
{

    double totalScalarProduct = 0.0;
    for (const auto& entry : partialError)
    {
        totalScalarProduct+= getPartialError(entry.first)*(k.getScalarProduct(entry.first.getFeatureVector(),t.getFeatureVector()));
    }


    return heavisideFunction(totalScalarProduct);

}

//returns the record of the error a specific sample had caused during training
double PerceptronClassifier::getPartialError(const Sample &t)
{
    for (const auto& entry : partialError)
    {
        if(entry.first.getIdentifier()==t.getIdentifier())
        {
            return entry.second;
        }
    }
    return 0.0;
}

//increases the record of the error a specific sample had caused during training
void PerceptronClassifier::incrementPartialError(const Sample &t, const double e)
{
    for (auto& entry : partialError)
    {
        if (entry.first.getIdentifier() == t.getIdentifier())
        {
            entry.second += e;
            return;
        }
    }
}

//checks if the classifier makes no error(overfitting)
bool PerceptronClassifier::checkIfClassifierIsPerfect()
{
    for (const auto& entry : partialError)
    {
        if (entry.first.getLabel() != classify(entry.first))
        {
            return false;
        }
    }

    return true;
}

double PerceptronClassifier::heavisideFunction(const double &x)
{
    if (x >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int PerceptronClassifier::getIterations()
{
    return iterations;
}

kernel& PerceptronClassifier::getUsedKernel()
{
    return k;
}

double PerceptronClassifier::calculateAccuracy(const std::vector<Sample> &testset)
{
    double correctCounter = 0;
    for(int i =0;i<testset.size();i++)
    {
        if(classify(testset[i]) == testset[i].getLabel())
        {
            correctCounter++;
        }
    }

    return correctCounter/testset.size();
}