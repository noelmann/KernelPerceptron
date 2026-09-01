#include "kernelmatrix.h"
#include <iostream>
#include <omp.h>

KernelMatrix::KernelMatrix(kernel &k) : usedKernel(k)
{
}

void KernelMatrix::calculateKernelMatrix(std::vector<Sample> samples,int maxKernelMatrixSize)
{
    int expectedKernelMatrixSize = (samples.size()*samples.size()+samples.size())/2;

    this->samples = samples;

    int matrixSampleSize = this->samples.size();

    if(expectedKernelMatrixSize > maxKernelMatrixSize)
    {
        matrixSampleSize = -0.5+sqrt(0.25+2*maxKernelMatrixSize);
    }

    matrix.resize(matrixSampleSize);

#pragma omp parallel for
    for(int y = 0;y<matrixSampleSize;y++)
    {
        matrix[y].resize(y+1);
        for(int x = 0;x<y+1;x++)
        {
            matrix[y][x] = (usedKernel.getScalarProduct(this->samples[y].getFeatureVector(),this->samples[x].getFeatureVector()));

            //std::cout << matrix[y][x] << " ";
        }

       // std::cout << std::endl;
    }

    determineTrainingSamplesKernelMatrixIndices(this->samples);




    /*for(int j = 0; j<samples.size();j++)
    {
        for(int k = 0;k<j+1;k++)
        {
            std::cout << matrix[y][x] << " ";
        }
    }*/
}

double KernelMatrix::getKernelMatrixEntry(int x, int y)
{   
    if(x > y)
    {
        return matrix[x][y];
    }
    else
    {
        return matrix[y][x];
    }

}

int KernelMatrix::getIndexOfSample(Sample &sample)
{
    for(int i = 0;i<this->samples.size();i++)
    {
        if(this->samples[i].getIdentifier() == sample.getIdentifier())
        {
            return i;
        }
    }

    return -1;
}

void KernelMatrix::determineTrainingSamplesKernelMatrixIndices(std::vector<Sample> &trainingSamples)
{
    trainingSampleKernelMatrixIndices.clear();
    for (int i = 0;i<trainingSamples.size();i++)
    {
        trainingSampleKernelMatrixIndices[trainingSamples[i].getIdentifier()]=getIndexOfSample(trainingSamples.at(i));
    }
}

int KernelMatrix::getKernelMatrixTrainingSampleIndex(Sample &s)
{
    return trainingSampleKernelMatrixIndices[s.getIdentifier()];
}


kernel& KernelMatrix::getUsedKernel()
{
    return usedKernel;
}

int KernelMatrix::getKernelMatrixSize()
{
    return matrix.size();
}