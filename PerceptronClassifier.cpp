//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"
#include <cmath>

PerceptronClassifier::PerceptronClassifier(std::vector<Sample> trainingSamples, bool useRBFkernel)
{
    useRBF= useRBFkernel;
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

        if (useRBF)
        {
            totalScalarProduct+= getPartialError(entry.first)*getRBFkernelScalarProduct(entry.first,t);
        }
        else
        {
            for (int i = 0;i<entry.first.getFeatureVector().size();i++)
            {
                totalScalarProduct+=getPartialError(entry.first)*entry.first.getFeatureVector()[i]*t.getFeatureVector()[i];
            }
        }
    }

    if (totalScalarProduct >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//returns the record of the error a specific sample had caused during training
double PerceptronClassifier::getPartialError(Sample t)
{
    for (const auto& entry : partialError)
    {
        if (entry.first.getFeatureVector() == t.getFeatureVector())
            return entry.second;
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

//calculates the scalar product of the feature vectors of two samples in a higher dimension using the RBF kernel
double PerceptronClassifier::getRBFkernelScalarProduct(Sample s1, Sample s2)
{
    double squaredEuclideanDistance = 0.0;
    double sigma = 1.0;
    for (int i = 0;i<s1.getFeatureVector().size();i++)
    {
        squaredEuclideanDistance += pow((s1.getFeatureVector()[i]-s2.getFeatureVector()[i]),2);
    }

    squaredEuclideanDistance/=2*pow(sigma,2);

    return (exp(-squaredEuclideanDistance));
}
