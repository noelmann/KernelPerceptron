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
    static std::vector<Entry> generateHyperparameterSet(std::vector<Entry> a, int hyperParameterVariableCount);

private:
    static std::vector<Entry> generateCartesianProduct(std::vector<Entry> a, std::vector<Entry> b);
};

#endif // HYPERPARAMETERGENERATOR_H
