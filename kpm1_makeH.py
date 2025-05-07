import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss

# buld Hamiltonian for graphene
# honeycomb lattice with periodic boundary conditions

def build_H(N1,N2):
    t0=2.7
    # same as in check_lattice.py
    def get_neighbors(m1,n1,sublattice,N1,N2):
        assert sublattice =="A" or sublattice == "B"
        ind_neighbors=[]
        if sublattice == "A":
            dmdn=[[0,0],[-1,0],[0,-1]]
        elif sublattice == "B":
            dmdn=[[0,0],[1,0],[0,1]]
        for i in range(len(dmdn)):
            m2=m1+dmdn[i][0]
            n2=n1+dmdn[i][1]
            ## OBC
            # if m2>=0 and m2<N1 and n2>=0 and n2<N2:
            #     ind2=(m2)*N2+n2
            #     ind_neighbors.append(ind2)
            ## PBC
            ind2=(m2%N1)*N2+n2%N2
            ind_neighbors.append(ind2)
        return ind_neighbors
    dim_H=2*N1*N2
    row=[]
    col=[]
    data=[]
    for m in range(N1):
        for n in range(N2):
            ind1A=m*N2+n
            ind1B=N1*N2+m*N2+n
            # nearest neighbors of A are sublattice B
            ind2A_list=get_neighbors(m,n,"A",N1,N2)
            for i2A in range(len(ind2A_list)):
                row.append(ind1A)
                col.append(N1*N2+ind2A_list[i2A])
                data.append(t0)
            # nearest neighbors of B are sublattice A
            ind2B_list=get_neighbors(m,n,"B",N1,N2)
            for i2B in range(len(ind2B_list)):
                row.append(ind1B)
                col.append(ind2B_list[i2B])
                data.append(t0)
    H_big=ss.csr_matrix((data, (row, col)),
                        shape=(dim_H, dim_H),dtype=complex)
    #Emin=np.min(ss.linalg.eigsh(H_big,k=6,which="SA")[0])
    #Emax=np.max(ss.linalg.eigsh(H_big,k=6,which="LA")[0])
    #print(Emin,Emax)
    Emin=-8.11; Emax=8.11
    eb=(Emax+Emin)*0.5
    ea=(Emax-Emin)/(2-0.01)
    # rescaled, eigvalues in (-1,1)
    print(f"a={ea}, b={eb}")
    H_tilde=(( H_big-eb*ss.eye(H_big.shape[0],format='csr',dtype=complex) )/ea)
    return H_tilde,Emin,Emax

















