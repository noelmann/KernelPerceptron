//
// Created by DerWiggler01 on 21.03.2026.
//

#ifndef KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#define KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#include "Sample.h"
#include "kernel.h"
#include <unordered_map>


class PerceptronClassifier
{
public:
    PerceptronClassifier(std::vector<Sample> trainingSamples, const kernel &kernel);
    double classify(Sample t);
    std::vector<std::pair<Sample,double>> partialError;
private:
    double heavisideFunction(const double &x);
    double getPartialError(Sample t);
    void incrementPartialError(Sample t, double e);
    bool checkIfClassifierIsPerfect();
    const kernel &k;

};


#endif //KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H