#include "kernel_rbf.h"

Kernel_RBF::Kernel_RBF(const double &sigma)
{
    setSigma(sigma);
}

double Kernel_RBF::getSigma()
{
    return s;
}
void Kernel_RBF::setSigma(const double &sigma)
{
    s = sigma;
}
double Kernel_RBF::getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2)
{
    double squaredEuclideanDistance = 0.0;
    for (int i = 0;i<v1.size();i++)
    {
        squaredEuclideanDistance += pow((v1[i]-v2[i]),2);
    }

    squaredEuclideanDistance/=2*pow(s,2);

    return (exp(-squaredEuclideanDistance));
}