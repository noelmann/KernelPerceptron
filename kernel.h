#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <cmath>

class kernel
{
public:
    virtual ~kernel() = default;
    virtual double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const = 0;
};

#endif // KERNEL_H
