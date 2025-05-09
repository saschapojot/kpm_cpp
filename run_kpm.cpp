#include "./cpp_subroutines/kpm_routine.hpp"
// using namespace arma;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }

    auto kpm_obj=kpm_computation(std::string(argv[1]));
    kpm_obj.build_H_sparse();
    arma::cx_dvec r_ket=arma::cx_dvec(2*kpm_obj.N1*kpm_obj.N2,arma::fill::zeros);
    r_ket(0)=std::complex<double>(1.0,0);
    r_ket(2)=std::complex<double>(1.0,0);
    arma::dvec moments;
    int Nm=10;
    kpm_obj.write_moments(kpm_obj.H_tilde,Nm,r_ket,moments);
    kpm_obj.print_ptr(moments.memptr(),Nm);

    return 0;
}