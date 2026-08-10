#ifndef KERNEL_POLYNOMIAL_H
#define KERNEL_POLYNOMIAL_H

#include <vector>
#include <cmath>

class kernel_polynomial
{

private:
    double n;

public:
    kernel_polynomial(const int &degree);
    double getDegree();
    void setDegree(const int &degree);
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2);
};

#endif // KERNEL_POLYNOMIAL_H
