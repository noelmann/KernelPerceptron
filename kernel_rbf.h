#ifndef KERNEL_RBF_H
#define KERNEL_RBF_H

#include <vector>
#include <cmath>

class Kernel_RBF
{

private:
    double s;

public:
    Kernel_RBF(const double &sigma);
    double getSigma();
    void setSigma(const double &sigma);
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2);
};

#endif // KERNEL_RBF_H
