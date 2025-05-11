//
// Created by adada on 6/5/2025.
//

#include "kpm_routine.hpp"

///
/// @param m1 index in direction 1
/// @return m1%N1 in python
int kpm_computation::mod_direction1(const int& m1)
{
    return ((m1 % N1) + N1) % N1;
}


///
/// @param m2 index in direction 2
/// @return m2%N2 in python
int kpm_computation::mod_mod_direction2(const int& m2)
{
    return ((m2 % N2) + N2) % N2;
}

///
/// @param n1 unit cell index in direction 1
/// @param n2 unit cell index in direction 2
/// @return a vector of indices of B around A
std::vector<int> kpm_computation::get_neighbors_of_A(const int& n1, const int& n2)
{
    std::vector<int> B_vec;
    for (const auto& d_vec : this->d_vec_around_A)
    {
        int m1 = n1 + d_vec[0];
        int m2 = n2 + d_vec[1];
        int neighbor = mod_direction1(m1) * N2 + mod_mod_direction2(m2);
        B_vec.push_back(neighbor);
    }
    return B_vec;
}


///
/// @param n1 unit cell index in direction 1
/// @param n2 unit cell index in direction 2
/// @return a vector of indices of A around B
std::vector<int> kpm_computation::get_neighbors_of_B(const int& n1, const int& n2)
{
    std::vector<int> A_vec;
    for (const auto& d_vec : this->d_vec_around_B)
    {
        int m1 = n1 + d_vec[0];
        int m2 = n2 + d_vec[1];
        int neighbor = mod_direction1(m1) * N2 + mod_mod_direction2(m2);
        A_vec.push_back(neighbor);
    }
    return A_vec;
}

///
/// @param n1 unit cell index in direction 1
/// @param n2 unit cell index in direction 2
/// @return flattened index n1*N2+n2
int kpm_computation::flattened_ind(const int& n1, const int& n2)
{
    return n1 * N2 + n2;
}

///
/// @return a dense matrix for Hamiltonian
arma::cx_dmat kpm_computation::build_H_dense()
{
    int length = 2 * N1 * N2;
    int offset_AB = N1 * N2;
    arma::cx_dmat H_mat = arma::cx_dmat(length, length, arma::fill::zeros);
    // H_mat.print("H_mat:");
    std::cout << "H_mat has " << H_mat.n_rows << " rows, " << H_mat.n_cols << " columns." << std::endl;
    std::vector<int> A_to_B_row_ind_vec;
    std::vector<int> A_to_B_col_ind_vec;
    std::vector<int> A_neighbors_vec;
    // construct A --> B row and col indices
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int ind1A = flattened_ind(n1, n2);
            //neighbors of A are B, offset = N1*N2
            A_neighbors_vec = this->get_neighbors_of_A(n1, n2);
            for (const int& elem : A_neighbors_vec)
            {
                A_to_B_row_ind_vec.push_back(ind1A);
                A_to_B_col_ind_vec.push_back(offset_AB + elem);
            } //end for elem in A_neighbors_vec
        } //end for n2
    } //end for n1
    // for (int i=0;i<A_to_B_row_ind_vec.size();i++)
    // {
    //     std::cout<<"("<<A_to_B_row_ind_vec[i]<<", "<<A_to_B_col_ind_vec[i]<<")"<<std::endl;
    // }

    std::vector<int> B_to_A_row_ind_vec;
    std::vector<int> B_to_A_col_ind_vec;
    std::vector<int> B_neighbors_vec;
    // construct B --> A row and col indices
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int ind1B = offset_AB + flattened_ind(n1, n2);
            //neighbors of B are A, offset = N1*N2
            B_neighbors_vec = this->get_neighbors_of_B(n1, n2);
            for (const int& elem : B_neighbors_vec)
            {
                B_to_A_row_ind_vec.push_back(ind1B);
                B_to_A_col_ind_vec.push_back(elem);
            } //end for elem in B_neighbors_vec
        } //end for n2
    } //end for n1
    // for (int i=0;i<B_to_A_row_ind_vec.size();i++)
    // {
    //     std::cout<<"("<<B_to_A_row_ind_vec[i]<<", "<<B_to_A_col_ind_vec[i]<<")"<<std::endl;
    // }
    //construct matrix , A to B part
    for (int i = 0; i < A_to_B_row_ind_vec.size(); i++)
    {
        int row_val = A_to_B_row_ind_vec[i];
        int col_val = A_to_B_col_ind_vec[i];
        H_mat(row_val, col_val) = std::complex<double>(t0, 0);
    } //end for i
    //construct matrix , B to A part
    for (int i = 0; i < B_to_A_row_ind_vec.size(); i++)
    {
        int row_val = B_to_A_row_ind_vec[i];
        int col_val = B_to_A_col_ind_vec[i];
        H_mat(row_val, col_val) = std::complex<double>(t0, 0);
    } //end for i
    // H_mat.row(2).print();
    return H_mat;
}


///
///build sparse matrix Hamiltonian, E max, E min
void kpm_computation::build_H_sparse()
{
    const auto t_build_Start{std::chrono::steady_clock::now()};
    // int length=2*N1*N2;
    int offset_AB = N1 * N2;
    this->H_big = arma::sp_cx_dmat(length, length);
    std::vector<int> A_to_B_row_ind_vec;
    std::vector<int> A_to_B_col_ind_vec;
    std::vector<int> A_neighbors_vec;
    // construct A --> B row and col indices
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int ind1A = flattened_ind(n1, n2);
            //neighbors of A are B, offset = N1*N2
            A_neighbors_vec = this->get_neighbors_of_A(n1, n2);
            for (const int& elem : A_neighbors_vec)
            {
                A_to_B_row_ind_vec.push_back(ind1A);
                A_to_B_col_ind_vec.push_back(offset_AB + elem);
            } //end for elem in A_neighbors_vec
        } //end for n2
    } //end for n1

    std::vector<int> B_to_A_row_ind_vec;
    std::vector<int> B_to_A_col_ind_vec;
    std::vector<int> B_neighbors_vec;
    // construct B --> A row and col indices
    for (int n1 = 0; n1 < N1; n1++)
    {
        for (int n2 = 0; n2 < N2; n2++)
        {
            int ind1B = offset_AB + flattened_ind(n1, n2);
            //neighbors of B are A, offset = N1*N2
            B_neighbors_vec = this->get_neighbors_of_B(n1, n2);
            for (const int& elem : B_neighbors_vec)
            {
                B_to_A_row_ind_vec.push_back(ind1B);
                B_to_A_col_ind_vec.push_back(elem);
            } //end for elem in B_neighbors_vec
        } //end for n2
    } //end for n1
    //construct matrix , A to B part
    for (int i = 0; i < B_to_A_row_ind_vec.size(); i++)
    {
        int row_val = A_to_B_row_ind_vec[i];
        int col_val = A_to_B_col_ind_vec[i];
        this->H_big(row_val, col_val) = std::complex<double>(t0, 0);
    } //end for i
    //construct matrix , B to A part
    for (int i = 0; i < B_to_A_row_ind_vec.size(); i++)
    {
        int row_val = B_to_A_row_ind_vec[i];
        int col_val = B_to_A_col_ind_vec[i];
        this->H_big(row_val, col_val) = std::complex<double>(t0, 0);
    } //end for i

    // arma::cx_dmat H_big_dense(H_big);
    // H_big_dense.row(length-1).print();
    //check whether the matrix is Hermitian

    bool is_hermitian = H_big.is_hermitian();
    std::cout << "is_hermitian=" << is_hermitian << std::endl;
    if (is_hermitian == false)
    {
        std::cerr << "Matrix is non-Hermitian" << std::endl;
        std::exit(2);
    }
    //largest eigenvalue
    arma::cx_dvec eig_vals_large;
    arma::cx_dmat eig_vecs_large;
    //largest eigenvalue
    arma::eigs_gen(eig_vals_large, eig_vecs_large, this->H_big, 1, "lr");
    this->E_big_max = eig_vals_large(0).real();

    std::cout << "E_big_max=" << E_big_max << std::endl;

    //smallest eigenvalue
    arma::cx_dvec eig_vals_small;
    arma::cx_dmat eig_vecs_small;
    arma::eigs_gen(eig_vals_small, eig_vecs_small, this->H_big, 1, "sr");
    this->E_big_min = eig_vals_small(0).real();
    std::cout << "E_big_min=" << E_big_min << std::endl;

    //construct H_tilde
    this->eps = 0.01;
    this->b = (E_big_min + E_big_max) / 2.0;
    std::complex<double> b_complex = std::complex<double>(b, 0);
    this->a = (E_big_max - E_big_min) / (2.0 - eps);
    std::complex<double> a_complex = std::complex<double>(a, 0);

    std::cout << "a=" << a << ", b=" << b << std::endl;

    arma::sp_cx_dmat I_mat = arma::speye<arma::sp_cx_dmat>(length, length);

    this->H_tilde = (H_big - b_complex * I_mat) / a_complex;
    // H_tilde.print();


    const auto t_build_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_build_End - t_build_Start};
    std::cout << "Building H_tilde time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
}


///
/// @param m
/// @param Nm
/// @param lamb
/// @return coefficients of gn
double kpm_computation::kpm_gn(const int& m, const int& Nm, const double& lamb)
{
    double Nm_db = static_cast<double>(Nm);
    double m_db = static_cast<double>(m);
    double factor1 = 1.0 / (Nm_db + 1.0);
    double cos_part = (Nm_db - m_db + 1.0) * std::cos(PI * m_db / (Nm_db + 1.0));

    double sin_part = std::sin(PI * m_db / (Nm_db + 1.0)) / std::tan(PI / (Nm + 1.0));

    return (cos_part + sin_part) * factor1;
}

///
/// @param n
/// @param x
/// @return T_{n}(x)
double kpm_computation::T_cheb(const int& n, const double& x)
{
    double n_db = static_cast<double>(n);
    return std::cos(n_db * std::acos(x));
}


///
/// @param H_tilde rescaled Hamiltonian
/// @param Nm number of moments
/// @param r_ket random vector |r> (normalized)
/// @param moments moments computed from |r>
void kpm_computation::write_moments(const arma::sp_cx_dmat& H_tilde, const int& Nm,
                                    const arma::cx_dvec& r_ket, arma::dvec& moments)
{
    //integer division
    // if Nm is odd, then 2 * Nm_cal +1 = Nm
    // if Nm is even, then 2* Nm_cal = Nm
    //in this code we assume that Nm_cal is even
    int Nm_cal = Nm / 2;


    //vectors for iteration
    arma::cx_dvec zj_minus1 = r_ket; //|z0>
    arma::cx_dvec zj = H_tilde * zj_minus1; // |z1>
    std::complex<double> two = std::complex<double>(2.0, 0);
    arma::cx_dvec zj_plus1 = two * H_tilde * zj - zj_minus1; // |z2>

    //initialize mu0, mu1
    std::complex<double> mu0_complex = std::complex<double>(1, 0); //<r|r>
    std::complex<double> mu1_complex = arma::cdot(r_ket, zj); //<r|z1>

    // std::cout<<"mu0_complex="<<mu0_complex<<std::endl;
    // std::cout<<"mu1_complex="<<mu1_complex<<std::endl;
    moments[0] = mu0_complex.real();
    moments[1] = mu1_complex.real();

    for (int j = 1; j <= Nm_cal - 1; j++)
    {
        std::complex<double> mu2j = two * arma::cdot(zj, zj) - mu0_complex;
        std::complex<double> mu2j_plus1 = two * arma::cdot(zj, zj_plus1) - mu1_complex;
        moments[2 * j] = mu2j.real();
        moments[2 * j + 1] = mu2j_plus1.real();
        zj_minus1 = zj;
        zj = zj_plus1;
        zj_plus1 = two * H_tilde * zj - zj_minus1;
    } //end for j
}


///
/// this function precompute  g_coef_vec (kernel coefficients)
void kpm_computation::compute_g_coef_vec()
{
    const auto t_g_Start{std::chrono::steady_clock::now()};
    this->g_coef_vec.resize(this->Nm);
    for (int m = 0; m <= Nm - 1; m++)
    {
        g_coef_vec[m] = kpm_gn(m, Nm, lamb);
    } //end for m
    const auto t_g_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_g_End - t_g_Start};
    std::cout << "build g coefs time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
}


///
/// rescaled energy in (-1,1)
void kpm_computation::construct_E_tilde_vec()
{
    const auto t_E_tilde_Start{std::chrono::steady_clock::now()};
    double scale = 1.1;
    double start = this->E_big_min / scale;
    double end = this->E_big_max / scale;
    std::cout << "E tilde start=" << start << ", E tilde end=" << end << std::endl;
    double step = (end - start) / (static_cast<double>(Q) - 1.0);

    this->E_tilde_vec.reserve(Q);
    for (int i = 0; i < Q; i++)
    {
        E_tilde_vec.push_back(start + i * step);
    } //end for i
    //rescale E_tilde_vec
    for (double& val : E_tilde_vec)
    {
        val = (val - b) / a;
    } //end rescaling

    const auto t_E_tilde_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_E_tilde_End - t_E_tilde_Start};
    std::cout << "build E_tilde time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
}

///
/// compute [g0,2g1 T1,...,2g_{N-1}T_{N-1}]
void kpm_computation::construct_coef_of_moments()
{
    const auto t_coef_of_moments_Start{std::chrono::steady_clock::now()};
    this->coef_of_moments = arma::dmat(Q, Nm, arma::fill::zeros);
    for (int q = 0; q < Q; q++)
    {
        for (int m = 0; m < Nm; m++)
        {
            if (m == 0)
            {
                coef_of_moments(q, m) = g_coef_vec[0];
            } //end  if m==0
            else
            {
                coef_of_moments(q, m) = 2.0 * g_coef_vec[m] * this->T_cheb(m, this->E_tilde_vec[q]);
            } //end else
        } //end for column index m
    } //end for row index q

    const auto t_coef_of_moments_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_coef_of_moments_End - t_coef_of_moments_Start};
    std::cout << "build coef of moments time: " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
}

///
/// compute dos serially
void kpm_computation::compute_dos_serial()
{
    this->build_H_sparse();
    this->compute_g_coef_vec();
    this->construct_E_tilde_vec();
    this->construct_coef_of_moments();
    std::vector<double>rho_E_tilde_all_q_vec=this->rho_E_tilde_all_q();
    this->write_dos_2_csv(this->E_tilde_vec,rho_E_tilde_all_q_vec);
}


///
/// @param r index of random vector
/// @param q index of E_tilde
/// @return rho_{r}(E_tilde)
double kpm_computation::rho_r_E_tilde(const int& r, const int& q)
{
    //random vector
    arma::cx_dvec r_ket = arma::randn<arma::cx_vec>(this->length);
    //normalize random vector
    r_ket = r_ket / std::complex<double>(arma::norm(r_ket, 2), 0);
    //initialize moments
    arma::dvec moments = arma::dvec(Nm, arma::fill::zeros);
    //compute moments
    this->write_moments(this->H_tilde, this->Nm, r_ket, moments);
    double rho_r_E_tilde_val = arma::dot(moments, this->coef_of_moments.row(q));
    return rho_r_E_tilde_val;
}


///
/// @param q index of E_tilde
/// @return rho(E_tilde)
double kpm_computation::rho_E_tilde(const int& q)
{
    const auto t_one_rho_E_tilde_Start{std::chrono::steady_clock::now()};
    double rho_E_tilde = 0;
    for (int r = 0; r < R + 1; r++)
    {
        rho_E_tilde += this->rho_r_E_tilde(r, q);
    } //end for r
    double E_tilde_tmp=this->E_tilde_vec[q];
    rho_E_tilde*=1.0/(PI*std::sqrt(1.0-std::pow(E_tilde_tmp,2.0)))*1.0/static_cast<double>(R);

    const auto t_one_rho_E_tilde_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_one_rho_E_tilde_End - t_one_rho_E_tilde_Start};
    std::cout << "rho_E_tilde time for q=" << q << ": " << elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;


    return rho_E_tilde;
}


///
///compute rho(E_tilde) for all q, and write to file
std::vector<double> kpm_computation::rho_E_tilde_all_q()
{
    const auto t_q_vec_Start{std::chrono::steady_clock::now()};
    std::vector<double> rho_E_tilde_all_q_vec;
    rho_E_tilde_all_q_vec.resize(Q);
    for (int q=0;q<Q;q++)
    {
        rho_E_tilde_all_q_vec[q]=rho_E_tilde(q);
    }//end for q
    const auto t_q_vec_End{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_secondsAll{t_q_vec_End - t_q_vec_Start};

    std::cout<<"all q time: "<< elapsed_secondsAll.count() / 3600.0 << " h" << std::endl;
    return rho_E_tilde_all_q_vec;
}

///
/// @param E_tilde_vec vector  of rescaled energy
/// @param rho_vec vector of dos
void kpm_computation::write_dos_2_csv(const std::vector<double>& E_tilde_vec,std::vector<double> & rho_vec)
{
    std::ofstream outFile(this->dataRoot+"/out_dos.csv");
    // Check if the file was opened successfully
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        std::exit(2);
    }
    outFile<<"E_tilde,dos\n";
    for (int j=0;j<E_tilde_vec.size();j++)
    {
        outFile<<E_tilde_vec[j]<<","<<rho_vec[j]<<"\n";
    }
    // Close the file
    outFile.close();
    std::cout << "Vectors have been written to file successfully." << std::endl;

}