#include "kernel_sigmoid.h"

kernel_sigmoid::kernel_sigmoid(const double &slope, const double &intercept)
{
    setSlope(slope);
    setIntercept(intercept);
}

double kernel_sigmoid::getIntercept()
{
    return c;
}

void kernel_sigmoid::setIntercept(const double &intercept)
{
    c = intercept;
}

double kernel_sigmoid::getSlope()
{
    return a;
}

void kernel_sigmoid::setSlope(const double &slope)
{
    a = slope;
}

//calculates the scalar product of the feature vectors of two samples in a higher dimension using the Sigmoid kernel
double kernel_sigmoid::getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const
{
    double scalarProduct = 0.0;
    for(int i =0;i<v1.size();i++)
    {
        scalarProduct+=(v1[i]*v2[i]);
    }

    scalarProduct*=a;

    scalarProduct+=c;

    return tanh(scalarProduct);
}
