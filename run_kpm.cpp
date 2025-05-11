#include "./cpp_subroutines/kpm_routine.hpp"
using namespace arma;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }

    auto kpm_obj=kpm_computation(std::string(argv[1]));
    kpm_obj.compute_dos_serial();


    return 0;
}