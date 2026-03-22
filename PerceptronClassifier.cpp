//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"

PerceptronClassifier::PerceptronClassifier(std::vector<Sample> trainingSamples)
{
    for (const auto& trainingSample : trainingSamples)
    {
        partialError[trainingSample] = 0.0;
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

double PerceptronClassifier::classify(Sample t)
{
    double classification;
    double totalScalarProduct;
    for (const auto& entry : partialError)
    {
        //classification+=entry.second*

        for (int i = 0;i<entry.first.getFeatureVector().size();i++)
        {
            totalScalarProduct+=getPartialError(entry.first)*entry.first.getFeatureVector()[i]*t.getFeatureVector()[i];
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

double PerceptronClassifier::getPartialError(Sample t)
{
    return partialError[t];
}
void PerceptronClassifier::incrementPartialError(Sample t, double e)
{
    partialError[t] += e;
}

bool PerceptronClassifier::checkIfClassifierIsPerfect()
{
    for (const auto& entry : partialError)
    {
        if (entry.first.getLabel() != classify(entry.first))
        {
            return false;
        }
    }
}