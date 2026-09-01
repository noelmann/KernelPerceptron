cmake_minimum_required(VERSION 4.1)
project(KernelPerceptron)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(KernelPerceptron main.cpp
        Sample.cpp
        Sample.h
        PerceptronClassifier.cpp
        PerceptronClassifier.h
        kernel_rbf.h kernel_rbf.cpp
        kernel_polynomial.h kernel_polynomial.cpp
        kernel_linear.h kernel_linear.cpp
        kernel.h
        kernel_sigmoid.h kernel_sigmoid.cpp
        hyperparametergenerator.h hyperparametergenerator.cpp
        kernelmatrix.h kernelmatrix.cpp
        classifierbuilder.h classifierbuilder.cpp
)

target_link_libraries(KernelPerceptron PRIVATE OpenMP::OpenMP_CXX)
