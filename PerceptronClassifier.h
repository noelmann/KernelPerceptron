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
    PerceptronClassifier(kernel &kernel);
    double classify(Sample t);
    void train(const std::vector<Sample> &trainingSamples, const int &max_iterations);
    void hyperParameterGridSearch(const std::vector<Sample> &tuningSamples, const int &max_iterations, double lowerBound, double upperBound, int steps);
    std::vector<std::pair<Sample,double>> partialError;
    int getIterations();
    double calculateAccuracy(const std::vector<Sample> &testset);
    kernel& getUsedKernel();
private:
    double heavisideFunction(const double &x);
    double getPartialError(Sample t);
    void incrementPartialError(Sample t, double e);
    bool checkIfClassifierIsPerfect();
    kernel &k;
    int iterations = 0;

};


#endif //KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H