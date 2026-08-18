#ifndef HYPERPARAMETERGENERATOR_H
#define HYPERPARAMETERGENERATOR_H

#include <vector>

struct Entry
{
    std::vector<double> values;
};

class HyperparameterGenerator
{
public:
    static std::vector<Entry> generateHyperparameterSet(int hyperParameterVariableCount , double lowerBound, double upperBound, double stepSize);

private:
    static std::vector<Entry> generateCartesianProduct(std::vector<Entry> a, std::vector<Entry> b);
};

#endif // HYPERPARAMETERGENERATOR_H
