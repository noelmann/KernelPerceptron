//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"
#include <cmath>

PerceptronClassifier::PerceptronClassifier(std::vector<Sample> trainingSamples, const kernel &kernel) : k(kernel)
{
    for (const auto& trainingSample : trainingSamples)
    {
        partialError.push_back({trainingSample, 0.0});
    }

    while (!checkIfClassifierIsPerfect())
    {
        for (const auto& trainingSample : trainingSamples)
        {
            double error = (trainingSample.getLabel()-classify(trainingSample));
            incrementPartialError(trainingSample,error);
        }
    }
}

//classifies a given sample based on either a normal perceptron in its dual form or in combination with an RBF kernel to handle nonlinearity
double PerceptronClassifier::classify(Sample t)
{

    double totalScalarProduct = 0.0;
    for (const auto& entry : partialError)
    {
        totalScalarProduct+= getPartialError(entry.first)*(k.getScalarProduct(entry.first.getFeatureVector(),t.getFeatureVector()));
    }

    return heavisideFunction(totalScalarProduct);
}

//returns the record of the error a specific sample had caused during training
double PerceptronClassifier::getPartialError(Sample t)
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
void PerceptronClassifier::incrementPartialError(Sample t, double e)
{
    for (auto& entry : partialError)
    {
        if (entry.first.getFeatureVector() == t.getFeatureVector())
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