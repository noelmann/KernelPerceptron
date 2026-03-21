//
// Created by DerWiggler01 on 18.03.2026.
//

#include "TrainingSample.h"


TrainingSample::TrainingSample(const std::vector<double>& f, double l)
{
    label = l;
    features = f;
}

double TrainingSample::getLabel() const
{
    return label;
}

std::vector<double> TrainingSample::getFeatureVector()
{
    return features;
}
