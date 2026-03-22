cmake_minimum_required(VERSION 4.1)
project(KernelPerceptron)

set(CMAKE_CXX_STANDARD 20)

add_executable(KernelPerceptron main.cpp
        Sample.cpp
        Sample.h
        PerceptronClassifier.cpp
        PerceptronClassifier.h)
