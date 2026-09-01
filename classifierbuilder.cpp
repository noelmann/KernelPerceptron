#include "classifierbuilder.h"
#include "Sample.h"
#include "KernelMatrix.h"
#include <map>
#include "PerceptronClassifier.h"
#include <iostream>
#include <vector>

ClassifierBuilder::ClassifierBuilder(std::vector <Sample> trainingSet, kernel &k, int maxTrainingIterations, int maxKernelMatrixSize) : kMatrix(k)
{
    this->maxTrainingIterations = maxTrainingIterations;
    kMatrix.calculateKernelMatrix(trainingSet, maxKernelMatrixSize);
    if(isMulticlassClassificationTask(trainingSet))
    {
        buildMulticlassClassifier(trainingSet);
    }
    else
    {
        buildBinaryClassifier(trainingSet);
    }
}

bool ClassifierBuilder::isMulticlassClassificationTask(std::vector <Sample> &trainingSet)
{
    std::map<double,int> labels;
    for(const auto &s : trainingSet)
    {

        if(labels.count(s.getLabel()) > 0)
        {
            continue;
        }

        labels[s.getLabel()];

        if(labels.size() > 2)
        {
            return true;
        }
    }

    if(labels.size() > 2)
    {
        return true;
    }

    return false;
}

void ClassifierBuilder::buildBinaryClassifier(std::vector <Sample> &trainingSet)
{
    PerceptronClassifier classifier = PerceptronClassifier(kMatrix,trainingSet[0].getLabel());
    classifier.train(trainingSet,maxTrainingIterations);
    classifiers.emplace_back(classifier);
}

void ClassifierBuilder::buildMulticlassClassifier(std::vector <Sample> &trainingSet)
{
    std::cout << "build multi" << std::endl;
    std::vector<double> classes = findAllClassesInDataset(trainingSet);
    std::cout << "gathered class labels" << std::endl;

#pragma omp parallel for
    for(int c = 0;c<classes.size();c++)
    {
        PerceptronClassifier perceptron = PerceptronClassifier(kMatrix,classes[c]);
        perceptron.train(trainingSet,maxTrainingIterations);
#pragma omp critical
        {
        classifiers.emplace_back(perceptron);
            std::cout << "Training completed for class:" << classes[c] << std::endl;
        }
    }

}

std::vector<PerceptronClassifier> ClassifierBuilder::getTrainedClassifiers()
{
    return classifiers;
}

std::vector<double> ClassifierBuilder::findAllClassesInDataset(std::vector <Sample> &dataset)
{
    std::vector<double> classes;
    std::map<double,int> c;
    for(auto &s: dataset)
    {
        c[s.getLabel()];
    }

    for(auto &pair : c)
    {
        classes.emplace_back(pair.first);
    }
    return classes;
}

double ClassifierBuilder::classifySample(Sample &s)
{
    if(classifiers.size() == 1)
    {
        return classifiers[0].classify(s);
    }
    else
    {
        double maxDistanceFromDecisionBoundary = -100000;
        double label;
        for(int p = 0;p<classifiers.size();p++)
        {
            double d = classifiers[p].classify(s,false);
            if(d > maxDistanceFromDecisionBoundary)
            {
                maxDistanceFromDecisionBoundary = d;
                label = classifiers[p].getPositiveClassLabel();

            }
        }

        return label;
    }
}

double ClassifierBuilder::calculateAccuracy(std::vector <Sample> &testSet)
{
    int correct = 0;

#pragma omp parallel for reduction(+:correct)
    for(int i = 0;i<testSet.size();i++)
    {
        if(classifySample(testSet[i]) == testSet[i].getLabel())
        {
            correct++;
        }
    }

    return (double)correct/double(testSet.size());
}