#ifndef KERNEL_LINEAR_H
#define KERNEL_LINEAR_H

#include "kernel.h"

class kernel_linear : public kernel
{
public:
    kernel_linear();
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const override;
    std::vector<double> getParameterVector() const override;
    void setParameterVector(std::vector<double> newParameters) override;
};

#endif // KERNEL_LINEAR_H
