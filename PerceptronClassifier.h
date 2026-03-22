//
// Created by DerWiggler01 on 21.03.2026.
//

#ifndef KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#define KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#include "Sample.h"
#include <unordered_map>


class PerceptronClassifier
{
public:
    PerceptronClassifier(std::vector<Sample> trainingSamples);
    double classify(Sample t);
    std::unordered_map<Sample,double> partialError;
private:
    double getPartialError(Sample t);
    void incrementPartialError(Sample t, double e);
    bool checkIfClassifierIsPerfect();

};


#endif //KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H