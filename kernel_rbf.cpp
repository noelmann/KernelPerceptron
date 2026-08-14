#include "kernel_rbf.h"

kernel_rbf::kernel_rbf(const double &sigma)
{
    setSigma(sigma);
}

double kernel_rbf::getSigma()
{
    return s;
}
void kernel_rbf::setSigma(const double &sigma)
{
    s = sigma;
}

//calculates the scalar product of the feature vectors of two samples in a higher dimension using the RBF kernel
double kernel_rbf::getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const
{
    double squaredEuclideanDistance = 0.0;
    for (int i = 0;i<v1.size();i++)
    {
        squaredEuclideanDistance += pow((v1[i]-v2[i]),2);
    }

    squaredEuclideanDistance/=2*pow(s,2);

    return (exp(-squaredEuclideanDistance));
}

std::vector<double> kernel_rbf::getParameterVector() const
{
    std::vector<double> vec;
    vec.emplace_back(s);

    return vec;
}

void kernel_rbf::setParameterVector(std::vector<double> newParameters)
{
    setSigma(newParameters[0]);
}
