//
// Created by DerWiggler01 on 18.03.2026.
//

#include "Sample.h"

//holds the label and the feature vector of a sample
Sample::Sample(const std::vector<double>& f, double l)
{
    label = l;
    features = f;
}

double Sample::getLabel() const
{
    return label;
}

const std::vector<double>& Sample::getFeatureVector() const
{
    return features;
}
