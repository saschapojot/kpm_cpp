cmake_minimum_required(VERSION 3.28)
project(kpm_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")


add_executable(run_kpm run_kpm.cpp cpp_subroutines/kpm_routine.cpp cpp_subroutines/kpm_routine.hpp)

target_compile_options(run_kpm PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Development)

# Find Boost
set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_ROOT "$ENV{HOME}/cppLocal/boostBuild")
find_package(Boost REQUIRED COMPONENTS python numpy filesystem)
# Armadillo configuration
find_path(ARMADILLO_INCLUDE_DIR armadillo HINTS "/home/adada/cppLocal/armadillo_intel/include")
find_library(ARMADILLO_LIBRARY armadillo HINTS "/home/adada/cppLocal/armadillo_intel/lib")

# Set specific optimization for critical files
set_source_files_properties(
        cpp_subroutines/kpm_routine.cpp
        run_kpm.cpp
        PROPERTIES COMPILE_OPTIONS "-O3"
)
# Include directories
target_include_directories(run_kpm PRIVATE
        ${ARMADILLO_INCLUDE_DIR}
        ${Python3_INCLUDE_DIRS}
)


# Link libraries
target_link_libraries(run_kpm PRIVATE
        pthread
        m
        ${ARMADILLO_LIBRARY}
        Boost::filesystem
        Boost::python
        Boost::numpy
        Python3::Python
)