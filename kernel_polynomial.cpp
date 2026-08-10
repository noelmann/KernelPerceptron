#include "kernel_polynomial.h"

kernel_polynomial::kernel_polynomial(const int &degree)
{
    setDegree(degree);
}

double kernel_polynomial::getDegree()
{
    return n;
}
void kernel_polynomial::setDegree(const int &degree)
{
    n = degree;
}

//calculates the scalar product of the feature vectors of two samples in a higher dimension using a Polynomial kernel
double kernel_polynomial::getScalarProduct(const std::vector<double> &v1, const std::vector<double> &v2) const
{
    double scalarProduct = 0.0;
    for(int i =0;i<v1.size();i++)
    {
        scalarProduct+=(v1[i]*v2[i]);
    }

    scalarProduct+=1;

    return pow(scalarProduct,n);
}
