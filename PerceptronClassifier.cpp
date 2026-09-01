//
// Created by DerWiggler01 on 21.03.2026.
//

#include "PerceptronClassifier.h"
#include <cmath>
#include "hyperparametergenerator.h"
#include <iostream>
#include <omp.h>


/*PerceptronClassifier::PerceptronClassifier(kernel &kernel, bool preComputeKernelMatrix) : k(kernel)
{
    partialError.clear();
    iterations = 0;
    useKernelMatrix = preComputeKernelMatrix;
}*/

PerceptronClassifier::PerceptronClassifier(KernelMatrix &kernelMatrix, double positiveClassLabel) : usedKernelMatrix(kernelMatrix)
{
    this->positiveClassLabel = positiveClassLabel;
    partialError.clear();
    iterations = 0;
}

void PerceptronClassifier::train(std::vector<Sample> &trainingSamples, int max_iterations)
{
    partialError.clear();
    iterations = 0;
    for (int i = 0;i<trainingSamples.size();i++)
    {
        partialError.push_back({trainingSamples[i], 0.0});
    }

    while (iterations < max_iterations)
    {
        bool perfect = true;
        double totalError = 0.0;

        double error;
        for (int e=0;e<partialError.size();e++)
        {
            if(positiveClassLabel == partialError[e].first.getLabel())
            {
                error = (1-classify(partialError[e].first,true, true));
            }
            else
            {
                error = (0-classify(partialError[e].first,true, true));
            }

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

/*void PerceptronClassifier::train(const std::vector<Sample> &trainingSamples, const int &max_iterations)
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
}*/

/*void PerceptronClassifier::hyperParameterGridSearch(const std::vector<Sample> &trainingSamples,const std::vector<Sample> &developmentSamples, const int &max_iterations, double lowerBound, double upperBound, double stepSize)
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
}*/


//classifies a given sample using a dual form perceptron in combination with a given kernel(linear/polynomial/rbf)

double PerceptronClassifier::classify(Sample &t, bool useActivationFunction, bool trainingPhase)
{

    double totalScalarProduct = 0.0;
//#pragma omp parallel for reduction(+:totalScalarProduct)
    for (int i =0;i<partialError.size();i++)
    {
        if(partialError[i].second == 0)
        {
            continue;
        }

        int classificationSample_index = usedKernelMatrix.getKernelMatrixTrainingSampleIndex(t);
        int currentTrainingSample_index = usedKernelMatrix.getKernelMatrixTrainingSampleIndex(partialError[i].first);

        if(trainingPhase && classificationSample_index < usedKernelMatrix.getKernelMatrixSize() && currentTrainingSample_index < usedKernelMatrix.getKernelMatrixSize())
        {
            //std::cout << "Used kernel matrix" << std::endl;

            totalScalarProduct += partialError[i].second*usedKernelMatrix.getKernelMatrixEntry(currentTrainingSample_index,classificationSample_index);
        }
        else
        {
            //std::cout << "Did not use kernel matrix" << std::endl;
            totalScalarProduct+= partialError[i].second*usedKernelMatrix.getUsedKernel().getScalarProduct(t.getFeatureVector(),partialError[i].first.getFeatureVector());
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

/*double PerceptronClassifier::classify(const Sample &t, const bool &useActivationFunction)
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

}*/

//returns the record of the error a specific sample had caused during training
/*double PerceptronClassifier::getPartialError(const Sample &t)
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
}*/

double PerceptronClassifier::heavisideFunction(double &x)
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

double PerceptronClassifier::getPositiveClassLabel()
{
    return positiveClassLabel;
}


//calculates the accuracy of a single perceptron on a given testset
double PerceptronClassifier::calculateAccuracy(std::vector<Sample> &testset)
{
    double correctCounter = 0;
    #pragma omp parallel for reduction(+:correctCounter)
    for(int i =0;i<testset.size();i++)
    {
        if(classify(testset[i]) == testset[i].getLabel())
        {
            correctCounter++;
        }
    }

    return correctCounter/testset.size();
}