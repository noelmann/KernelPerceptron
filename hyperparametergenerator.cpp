#include "hyperparametergenerator.h"

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

std::vector<Entry> HyperparameterGenerator::generateHyperparameterSet(std::vector<Entry> a, int hyperParameterVariableCount)
{
    std::vector<Entry> entries = a;
    for(int i=1;i<hyperParameterVariableCount;i++)
    {
        entries = generateCartesianProduct(a,entries);
    }

    return entries;
}