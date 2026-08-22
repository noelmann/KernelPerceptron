//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"
#include <cmath>
#include "hyperparametergenerator.h"

PerceptronClassifier::PerceptronClassifier(kernel &kernel, bool preComputeKernelMatrix) : k(kernel)
{
    partialError.clear();
    iterations = 0;
    useKernelMatrix = preComputeKernelMatrix;
}

void PerceptronClassifier::train(const std::vector<Sample> &trainingSamples, const int &max_iterations)
{
    partialError.clear();
    if(useKernelMatrix)
    {
        calculateKernelMatrix(trainingSamples);
    }
    iterations = 0;
    for (int i = 0;i<trainingSamples.size();i++)
    {
        partialError.push_back({trainingSamples[i], 0.0});
    }

    while (iterations < max_iterations)
    {
        bool perfect = true;
        double totalError = 0.0;

        for (int e=0;e<partialError.size();e++)
        {
            double error = (partialError[e].first.getLabel()-classify(partialError[e].first,true));
            partialError[e].second += error;
            if(error != 0.0)
            {
                perfect = false;
            }
        }


            iterations++;

            if(perfect)
            {
                break;
            }

    }
}

void PerceptronClassifier::hyperParameterGridSearch(const std::vector<Sample> &trainingSamples,const std::vector<Sample> &developmentSamples, const int &max_iterations, double lowerBound, double upperBound, double stepSize)
{
    double bestAccuracy = -0.1;
    double acc = bestAccuracy;
    std::vector<double> bestVec;
    std::vector<double> vec;
    std::vector<Entry> hyperParameterSet = HyperparameterGenerator::generateHyperparameterSet(k.getParameterVector().size(),lowerBound,upperBound,stepSize);
    struct Entry standard;
    standard.values = k.getParameterVector();
    hyperParameterSet.insert(hyperParameterSet.begin(),standard);

    for(const struct Entry &e : hyperParameterSet)
    {
        vec.clear();

        for(const double &value : e.values)
        {
            vec.emplace_back(value);
        }

        k.setParameterVector(vec);

        train(trainingSamples,max_iterations);
        acc = calculateAccuracy(developmentSamples);

        if(acc > bestAccuracy)
        {
            bestAccuracy = acc;
            bestVec = vec;
        }
    }
    k.setParameterVector(bestVec);
    train(trainingSamples,max_iterations);
}


//classifies a given sample using a dual form perceptron in combination with a given kernel(linear/polynomial/rbf)
double PerceptronClassifier::classify(const Sample &t, const bool &useActivationFunction)
{

    double totalScalarProduct = 0.0;
    double matrixDimension = (sqrt(kernelMatrix.size()));

    for (int i =0;i<partialError.size();i++)
    {

        //looks up the scalar product in the kernel matrix if it exists and the matrix is to be used
        if (useKernelMatrix && partialError[i].first.getIdentifier() < matrixDimension && t.getIdentifier() < matrixDimension)
        {
            int index= partialError[i].first.getIdentifier()*matrixDimension+t.getIdentifier();
            totalScalarProduct+= partialError[i].second*kernelMatrix[index];
        }
        else
        {

            totalScalarProduct+= partialError[i].second*k.getScalarProduct(t.getFeatureVector(),partialError[i].first.getFeatureVector());
        }
    }



    if(useActivationFunction)
    {
        return heavisideFunction(totalScalarProduct);
    }
    else
    {
        return totalScalarProduct;
    }

}

//returns the record of the error a specific sample had caused during training
double PerceptronClassifier::getPartialError(const Sample &t)
{
    for (const auto& entry : partialError)
    {
        if(entry.first.getIdentifier()==t.getIdentifier())
        {
            return entry.second;
        }
    }
    return 0.0;
}

//increases the record of the error a specific sample had caused during training
void PerceptronClassifier::incrementPartialError(const Sample &t, const double e)
{
    for (auto& entry : partialError)
    {
        if (entry.first.getIdentifier() == t.getIdentifier())
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
        if (entry.first.getLabel() != classify(entry.first, true))
        {
            return false;
        }
    }

    return true;
}

double PerceptronClassifier::heavisideFunction(const double &x)
{
    if (x >= 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int PerceptronClassifier::getIterations()
{
    return iterations;
}

kernel& PerceptronClassifier::getUsedKernel()
{
    return k;
}

//precomputes the kernelMatrix in parallel
void PerceptronClassifier::calculateKernelMatrix(const std::vector<Sample> &trainingSamples)
{


    int max_i = trainingSamples.size();
    int max_j = trainingSamples.size();
    if(maxKernelMatrixSize < trainingSamples.size()*trainingSamples.size())
    {
        max_i=(sqrt(maxKernelMatrixSize));

        max_j=max_i;
    }

    kernelMatrix.resize(max_i*max_j);
#pragma omp parallel for
    for(int i = 0;i<max_i;i++)
    {
        for(int j = 0;j<max_j;j++)
        {

            int index = i*max_j+j;

            kernelMatrix[index]=k.getScalarProduct(trainingSamples[i].getFeatureVector(),trainingSamples[j].getFeatureVector());
        }
    }
}

//calculates the accuracy of a single perceptron on a given testset
double PerceptronClassifier::calculateAccuracy(const std::vector<Sample> &testset)
{
    double correctCounter = 0;
    for(int i =0;i<testset.size();i++)
    {
        if(classify(testset[i],true) == testset[i].getLabel())
        {
            correctCounter++;
        }
    }

    return correctCounter/testset.size();
}