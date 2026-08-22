#include "kernel_linear.h"

kernel_linear::kernel_linear()
{

}

double kernel_linear::getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const
{
    double scalarProduct = 0.0;
    for(int i =0;i<v1.size();i++)
    {
        scalarProduct+=(v1[i]*v2[i]);
    }
    return scalarProduct;
}

std::vector<double> kernel_linear::getParameterVector() const
{
    std::vector<double> vec;
    vec.emplace_back(0);
    return vec;
}

void kernel_linear::setParameterVector(std::vector<double> newParameters)
{
}
