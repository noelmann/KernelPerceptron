//
// Created by DerWiggler01 on 21.03.2026.
//

#ifndef KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#define KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H
#include "Sample.h"
#include "kernel.h"
#include <unordered_map>
#include <omp.h>
#include <map>
#include "KernelMatrix.h"

class PerceptronClassifier
{
public:
    PerceptronClassifier(KernelMatrix &kernelMatrix, double positiveClassLabel);

    double classify(Sample &t, bool useActivationFunction = true, bool trainingPhase = false);
    void train(std::vector<Sample> &trainingSamples, int max_iterations);

    //void hyperParameterGridSearch(const std::vector<Sample> &trainingSamples,const std::vector<Sample> &developmentSamples, const int &max_iterations, double lowerBound, double upperBound, double stepSize);
    std::vector<std::pair<Sample,double>> partialError;
    double getPositiveClassLabel();
    int getIterations();

    //kernel& getUsedKernel();
private:
    double heavisideFunction(double &x);
    double calculateAccuracy(std::vector<Sample> &testset);
    //double getPartialError(const Sample &t);
    //void incrementPartialError(const Sample &t, const double e);
    //bool checkIfClassifierIsPerfect();
    int iterations = 0;
    double positiveClassLabel;
    KernelMatrix &usedKernelMatrix;


};


#endif //KERNELPERCEPTRON_PERCEPTRONCLASSIFIER_H