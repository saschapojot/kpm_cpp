cpp code for kpm

###########################
python part
1. plot lattice, obtain neighbors of a unit cell
    python check_lattice.py
2. in kpm1_makeH.py, the Hamiltonian is built
3. the computation contains in kpm2_cheb.py

#############################
c++ part
1. initialize directories and parameters:
   python mk_dir.py
2. cmake .
3. make run_kpm
4. ./run_kpm ./path/to/cppIn.txt


#############################
post processing part:
1. cd plt
2. python load_csv_plt_dos.py