//
// Created by adada on 6/5/2025.
//

#ifndef KPM_ROUTINE_HPP
#define KPM_ROUTINE_HPP
#include <algorithm>
#include <armadillo>
#include <boost/filesystem.hpp>
// #include <boost/python.hpp>
// #include <boost/python/numpy.hpp>
#include <cfenv> // for floating-point exceptions
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = boost::filesystem;
// namespace py = boost::python;
// namespace np = boost::python::numpy;

constexpr double PI = M_PI;

class kpm_computation
{
public:

    kpm_computation(const std::string& cppInParamsFileName)
    {
        std::ifstream file(cppInParamsFileName);
        if (!file.is_open())
        {
            std::cerr << "Failed to open the file." << std::endl;
            std::exit(20);
        }
        std::string line;
        int paramCounter = 0;
        while (std::getline(file, line))
        {
            // Check if the line is empty
            if (line.empty())
            {
                continue; // Skip empty lines
            }
            std::istringstream iss(line);
            //read N
            if (paramCounter == 0)
            {
                iss>>N1;
                N2=N1;
                if (N1<=0)
                {
                    std::cerr << "N1 must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout<<"N1="<<N1<<", N2="<<N2<<std::endl;
                this->length=2*N1*N2;
                std::cout<<"length="<<length<<std::endl;
                paramCounter++;
                continue;
            }//end N

            //read Nm
            if (paramCounter == 1)
            {
                iss>>Nm;
                if (Nm<=0)
                {
                    std::cerr << "Nm must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout<<"Nm="<<Nm<<std::endl;
                paramCounter++;
                continue;
            }//end Nm
            //read R
            if (paramCounter == 2)
            {
                iss>>R;
                if (R<=0)
                {
                    std::cerr << "R must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout<<"R="<<R<<std::endl;
                paramCounter++;
                continue;
            }//end R
            //read lamb
            if (paramCounter == 3)
            {
                iss>>lamb;
                std::cout<<"lamb="<<lamb<<std::endl;
                paramCounter++;
                continue;
            }//end lamb
            //read t0
            if (paramCounter == 4)
            {
                iss>>t0;
                std::cout<<"t0="<<t0<<std::endl;
                paramCounter++;
                continue;
            }//end t0
        //read parallel_num
            if (paramCounter == 5)
            {
                iss>>parallel_num;
                if (parallel_num<=0)
                {
                    std::cerr << "parallel_num must be >0" << std::endl;
                    std::exit(1);
                }
                std::cout<<"parallel_num="<<parallel_num<<std::endl;
                paramCounter++;
                continue;
            }//end parallel_num
            //read Q
            if (paramCounter == 6)
            {
                iss>>Q;
                if (Q<=0)
                {
                    std::cerr<<"Q must be >0"<<std::endl;
                    std::exit(1);
                }
                std::cout<<"Q="<<Q<<std::endl;
                paramCounter++;
                continue;
            }//end Q

        }//end while
    this->d_vec_around_A={{0,0},{-1,0},{0,-1}};
        this->d_vec_around_B={{0,0},{1,0},{0,1}};
        boost::filesystem::path filePath(cppInParamsFileName);
        this->dataRoot=filePath.parent_path().string();
std::cout<<"dataRoot="<<dataRoot<<std::endl;
    }//end constructor

public:
    ///compute in parallel
    void compute_dos_parallel();
    ///initializes all r_ket
    void allocate_r_ket_all();
    ///
    /// @param E_tilde_vec vector  of rescaled energy
    /// @param rho_vec vector of dos
    void write_dos_2_csv(const std::vector<double>& E_tilde_vec,std::vector<double> & rho_vec);
    ///
    ///compute rho(E_tilde) for all q, and write to file
    std::vector<double>  rho_E_tilde_all_q();
    ///
    /// @param q index of E_tilde
    /// @return rho(E_tilde)
    double rho_E_tilde(const int &q);
    ///
    /// @param r index of random vector
    /// @param q index of E_tilde
    /// @return rho_{r}(E_tilde)
    double rho_r_E_tilde(const int &r, const int &q);
    ///
    /// compute [g0,2g1 T1,...,2g_{N-1}T_{N-1}]
    void construct_coef_of_moments();
    ///
    /// compute dos serially
    void compute_dos_serial();
    ///
    /// rescaled energy in (-1,1)
    void construct_E_tilde_vec();
    ///
    /// this function precompute  g_coef_vec (kernel coefficients)
    void compute_g_coef_vec();
    ///
    /// @param H_tilde rescaled Hamiltonian
    /// @param Nm number of moments
    /// @param r_ket random vector |r>  (normalized)
    /// @param moments moments computed from |r>
    void write_moments(const arma::sp_cx_dmat & H_tilde, const int &Nm, const arma::cx_dvec & r_ket, arma::dvec &moments);
    ///
    /// @param n
    /// @param x
    /// @return T_{n}(x)
    double T_cheb(const int &n, const double &x);
    ///
    /// @param m
    /// @param Nm
    /// @param lamb
    /// @return coefficients of gn
    double kpm_gn(const int &m, const int & Nm, const double &lamb);
    ///
    ///build sparse matrix Hamiltonian, E max, E min
    void build_H_sparse();
    /// 
    /// @return a dense matrix for Hamiltonian
    arma::cx_dmat build_H_dense();
    ///
    /// @param n1 unit cell index in direction 1
    /// @param n2 unit cell index in direction 2
    /// @return a vector of indices of A around B
    std::vector<int> get_neighbors_of_B(const int &n1, const int & n2);
    ///
    /// @param n1 unit cell index in direction 1
    /// @param n2 unit cell index in direction 2
    /// @return a vector of indices of B around A
    std::vector<int> get_neighbors_of_A(const int &n1, const int & n2);

    ///
    /// @param n1 unit cell index in direction 1
    /// @param n2 unit cell index in direction 2
    /// @return flattened index n1*N2+n2
    int flattened_ind(const int &n1, const int & n2);

    ///
    //    /// @param m1 index in direction 1
    //    /// @return m1%N1 in python
    int mod_direction1(const int&m1);

    ///
    /// @param m2 index in direction 2
    /// @return m2%N2 in python
    int mod_mod_direction2(const int&m2);


    // Template function to print the contents of a std::vector<T>
    template <typename T>
    void print_vector(const std::vector<T>& vec)
    {
        // Check if the vector is empty
        if (vec.empty())
        {
            std::cout << "Vector is empty." << std::endl;
            return;
        }

        // Print each element with a comma between them
        for (size_t i = 0; i < vec.size(); ++i)
        {
            // Print a comma before all elements except the first one
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << vec[i];
        }
        std::cout << std::endl;
    }
    template <class T>
    void print_shared_ptr(std::shared_ptr<T> ptr, const int& size)
    {
        if (!ptr)
        {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for (int i = 0; i < size; i++)
        {
            if (i < size - 1)
            {
                std::cout << ptr[i] << ",";
            }
            else
            {
                std::cout << ptr[i] << std::endl;
            }
        }
    } //end print_shared_ptr

    template <class T>
    void print_ptr(T*  ptr, const int& size)
    {
        if (!ptr)
        {
            std::cout << "Pointer is null." << std::endl;
            return;
        }

        for (int i = 0; i < size; i++)
        {
            if (i < size - 1)
            {
                std::cout << ptr[i] << ",";
            }
            else
            {
                std::cout << ptr[i] << std::endl;
            }
        }
    } //end print_ptr
public:
    std::vector<std::vector<int>>d_vec_around_A;// difference of position of B neighbors around A
    std::vector<std::vector<int>> d_vec_around_B;// difference of position of A neighbors around B
    int N1, N2;
    int Nm;
    double lamb;
    double t0;
    int R;
    int parallel_num;
    arma::sp_cx_dmat H_big;//unscaled Hamiltonian
    double E_big_max;
    double E_big_min;
    arma::sp_cx_dmat H_tilde;//scaled Hamiltonian
    double eps;
    std::vector<double>g_coef_vec;//kernel coefficients
    std::vector<double> E_tilde_vec;//vector of rescaled energy, in (-1,1)
    int Q;
    arma::dmat coef_of_moments;
    double a;
    double b;
    int length;
    std::string dataRoot;

    arma::cx_dcube  r_ket_all;//all r_ket


};
#endif //KPM_ROUTINE_HPP
