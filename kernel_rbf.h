#ifndef KERNEL_RBF_H
#define KERNEL_RBF_H

#include "kernel.h"

class kernel_rbf : public kernel
{

private:
    double s;

public:
    kernel_rbf(const double &sigma);
    double getSigma();
    void setSigma(const double &sigma);
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const override;
};

#endif // KERNEL_RBF_H
