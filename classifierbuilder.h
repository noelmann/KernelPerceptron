#ifndef CLASSIFIERBUILDER_H
#define CLASSIFIERBUILDER_H

#include <vector>
#include <map>
#include "Sample.h"
#include "kernel.h"
#include "kernel_linear.h"
#include "kernelmatrix.h"
#include "perceptronclassifier.h"

class ClassifierBuilder
{

private:
    bool isMulticlassClassificationTask(std::vector <Sample> &trainingSet);
    void buildBinaryClassifier(std::vector <Sample> &trainingSet);
    void buildMulticlassClassifier(std::vector <Sample> &trainingSet);
    KernelMatrix kMatrix;
    std::vector<PerceptronClassifier> classifiers;
    int maxTrainingIterations;
public:
    ClassifierBuilder(std::vector <Sample> trainingSet, kernel &k, int maxTrainingIterations = 100, int maxKernelMatrixSize = (3000*3000));
    std::vector<PerceptronClassifier> getTrainedClassifiers();
    std::vector<double> findAllClassesInDataset(std::vector <Sample> &dataset);
    double calculateAccuracy(std::vector <Sample> &testSet);
    double classifySample(Sample &s);

};

#endif // CLASSIFIERBUILDER_H
