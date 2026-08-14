#ifndef KERNEL_SIGMOID_H
#define KERNEL_SIGMOID_H

#include "kernel.h"
#include <cmath>

class kernel_sigmoid : public kernel
{
private:
    double a;
    double c;

public:
    kernel_sigmoid(const double &slope, const double &intercept);
    double getSlope();
    void setSlope(const double &slope);
    double getIntercept();
    void setIntercept(const double &intercept);
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const override;
    std::vector<double> getParameterVector() const override;
    void setParameterVector(std::vector<double> newParameters) override;
};

#endif // KERNEL_SIGMOID_H
