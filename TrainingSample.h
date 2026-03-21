//
// Created by DerWiggler01 on 18.03.2026.
//

#ifndef KERNELPERCEPTRON_TRAININGSAMPLE_H
#define KERNELPERCEPTRON_TRAININGSAMPLE_H

#include <vector>
class TrainingSample
{
    private:
    double label;
    std::vector<double> features;


    public:
    double getLabel() const;
    std::vector<double> getFeatureVector();
    TrainingSample(const std::vector<double>& features, double label);
};


#endif //KERNELPERCEPTRON_TRAININGSAMPLE_H