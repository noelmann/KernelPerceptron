#include "hyperparametergenerator.h"

//generates the cartesian product of two sets
std::vector<Entry> HyperparameterGenerator::generateCartesianProduct(std::vector<Entry> a, std::vector<Entry> b)
{
        std::vector<Entry> c;
        for(int i=0;i<a.size();i++)
        {
            for(int j=0;j<b.size();j++)
            {
                struct Entry newEntry;
                for(const auto &a_entryValue : a[i].values)
                {
                    newEntry.values.emplace_back(a_entryValue);
                }
                for(const auto &b_entryValue : b[j].values)
                {
                    newEntry.values.emplace_back(b_entryValue);
                }

                c.emplace_back(newEntry);
            }
        }

        return c;
}

//generates multiple hyperparameter configurations within a given range
std::vector<Entry> HyperparameterGenerator::generateHyperparameterSet(int hyperParameterVariableCount, double lowerBound, double upperBound, double stepSize)
{
    std::vector<Entry> entries_initial;

    int steps = (upperBound-lowerBound)/stepSize;

    for(int i = 0; i<=steps;i++)
    {
        struct Entry e;
        e.values.emplace_back(lowerBound+i*stepSize);
        entries_initial.emplace_back(e);
    }


    std::vector<Entry> entries = entries_initial;

    for(int i=1;i<hyperParameterVariableCount;i++)
    {
        entries = generateCartesianProduct(entries_initial,entries);
    }

    return entries;
}