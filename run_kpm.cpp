#include "./cpp_subroutines/kpm_routine.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }

    auto kpm_obj=kpm_computation(std::string(argv[1]));
    kpm_obj.build_H_sparse();
    // auto vec_B=kpm_obj.get_neighbors_of_A(1,1);
    // auto vec_A=kpm_obj.get_neighbors_of_B(1,2);
    // kpm_obj.print_vector(vec_B);
    // kpm_obj.print_vector(vec_A);
    return 0;
}