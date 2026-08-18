#ifndef KERNEL_POLYNOMIAL_H
#define KERNEL_POLYNOMIAL_H

#include "kernel.h"

class kernel_polynomial : public kernel
{

private:
    double n;

public:
    kernel_polynomial(const double &degree);
    double getDegree();
    void setDegree(const double &degree);
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const override;
    std::vector<double> getParameterVector() const override;
    void setParameterVector(std::vector<double> newParameters) override;
};

#endif // KERNEL_POLYNOMIAL_H
