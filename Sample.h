//
// Created by DerWiggler01 on 18.03.2026.
//

#ifndef KERNELPERCEPTRON_SAMPLE_H
#define KERNELPERCEPTRON_SAMPLE_H

#include <vector>
class Sample
{
    private:
    double label;
    std::vector<double> features;
    int identifier;


    public:
    double getLabel() const;
    int getIdentifier() const;
    const std::vector<double>& getFeatureVector() const;
    Sample(const int id, const std::vector<double> features, double label = 0);
};


#endif //KERNELPERCEPTRON_SAMPLE_H