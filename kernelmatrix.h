#ifndef KERNELMATRIX_H
#define KERNELMATRIX_H

#include "Sample.h"
#include "kernel.h"
#include <vector>
#include <map>

class KernelMatrix
{
private:
    std::vector<std::vector<double>>matrix;
    std::vector<Sample> samples;
    kernel &usedKernel;
    std::map<int,int> trainingSampleKernelMatrixIndices;
    void determineTrainingSamplesKernelMatrixIndices(std::vector<Sample> &trainingSamples);
    int getIndexOfSample(Sample &sample);

public:
    KernelMatrix(kernel &kernel);
    void calculateKernelMatrix(std::vector<Sample> samples, int maxKernelMatrixSize);
    double getKernelMatrixEntry(int x, int y);
    kernel& getUsedKernel();
    int getKernelMatrixTrainingSampleIndex(Sample &s);
    int getKernelMatrixSize();
};

#endif // KERNELMATRIX_H
