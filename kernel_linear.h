#ifndef KERNEL_LINEAR_H
#define KERNEL_LINEAR_H

class kernel_linear
{
public:
    kernel_linear();
    double getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2);
};

#endif // KERNEL_LINEAR_H
