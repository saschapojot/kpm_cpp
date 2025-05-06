//
// Created by adada on 6/5/2025.
//

#ifndef KPM_ROUTINE_HPP
#define KPM_ROUTINE_HPP
#include <boost/filesystem.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = boost::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;

constexpr double PI = M_PI;

class kpm_computation
{

public:
    int N1, N2;
    int Nm;
    double lamb;
    double t0;
    int R;
    int parallel_num;

};
#endif //KPM_ROUTINE_HPP
