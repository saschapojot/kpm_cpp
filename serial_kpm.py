import numpy as np
from numba import jit
import scipy.sparse as ss
import scipy.sparse.linalg as spla
from datetime import datetime
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
    # print(sorted(col))
    H_big=ss.csr_matrix((data, (row, col)),
                        shape=(dim_H, dim_H),dtype=complex)
    #Emin=np.min(ss.linalg.eigsh(H_big,k=6,which="SA")[0])
    #Emax=np.max(ss.linalg.eigsh(H_big,k=6,which="LA")[0])
    #print(Emin,Emax)
    # Emin=-8.11; Emax=8.11
    eigvals_small, eigvecs_small = spla.eigs(H_big, k=1, which='SR')
    Emin=np.real(eigvals_small[0])
    eigvals_large, eigvecs_large = spla.eigs(H_big, k=1, which='LR')
    Emax=np.real(eigvals_large[0])
    eb=(Emax+Emin)*0.5
    ea=(Emax-Emin)/(2-0.01)
    print(f"a={ea}, b={eb}")
    # rescaled, eigvalues in (-1,1)
    H_tilde=(( H_big-eb*ss.eye(H_big.shape[0],format='csr',dtype=complex) )/ea)
    return H_tilde,Emin,Emax,H_big.todense(),H_big




@jit(nopython=True)
def kpm(n,Nm2,para):
    poly=((Nm2-n+1)*np.cos(np.pi*n/(Nm2+1))+np.sin(np.pi*n/(Nm2+1))/np.tan(np.pi/(Nm2+1)))/(Nm2+1) #jackson
    #poly=np.sinh(para*(1-n/Nm2))/np.sinh(para) #lorentz
    #poly=1-n/Nm2 #Feje
    return poly
@jit(nopython=True)
def T_cheb(n,x): # put energy as input
    return np.cos(n*np.arccos(x))
def write_moments(H,Nm):
    # if Nm=10000, iterations are only done up to m=4999 (from 0 to 4999, inclusive)
    Nm_cal=round(Nm/2)
    # ea and eb will be saved with the moments, for calculating LDOS
    H0=np.array(np.random.normal(scale=1.0,size=(H.shape[0])),dtype=complex)
    H1=H.dot(H0)
    H_moments=np.zeros(Nm,dtype=np.double)
    H_moments[0]=np.real((H0.T.conj().dot(H0)))
    H_moments[1]=np.real((H0.T.conj().dot(H1)))
    for im in range(1,Nm_cal):
        H_temp=2*H.dot(H1)-H0
        H_moments[2*im]=np.real(2*(H1.T.conj().dot(H1))-H_moments[0])
        H_moments[2*im+1]=np.real(2*(H_temp.T.conj().dot(H1))-H_moments[1])
        H0=H1
        H1=H_temp
    return np.real(H_moments)
@jit(nopython=True)
def dos_kpm(e_list,Emin,Emax,moments,Nm2):
    lamb=1 #not used if it is not lorentz
    eb=(Emax+Emin)*0.5
    ea=(Emax-Emin)/(2-0.01)
    et_list=1/ea*e_list-1/ea*np.full(len(e_list),eb)#scaled energy
    print(et_list)
    d_list=np.zeros((len(et_list)))
    for iet in range(len(et_list)):
        #kpm (n,Nm2,para): values of g
        dos_recons=moments[0]*kpm(0,Nm2,lamb)+2*moments[1]*kpm(1,Nm2,lamb)*T_cheb(1,et_list[iet])
        for im in range(2,Nm2):
            mun=moments[im]
            dos_recons=dos_recons+2*mun*kpm(im,Nm2,lamb)*T_cheb(im,et_list[iet])
        d_list[iet]=1/np.sqrt(1-et_list[iet]**2)*dos_recons
    return d_list

def calculate_moment_and_dos(e_list,H,Emin,Emax,Nm):
    moments=write_moments(H,Nm)
    d_list=dos_kpm(e_list,Emin,Emax,moments,Nm)
    return d_list
R=5 # number of random vectors
Nm=50 # number of moments
B=0.0
N1=3
N2=3

H_tilde,Emin,Emax,H_big_dense,H_big=build_H(N1,N2)

print(f"N1={N1}, N2={N2}, Emin={Emin}, Emax={Emax}")
e_list=np.linspace(-6,6,1001)
# print(e_list)
d_list=np.zeros(len(e_list))

t_moments_dos_start= datetime.now()
calculate_moment_and_dos(e_list,H_tilde,Emin,Emax,Nm)

t_moments_dos_end= datetime.now()

print(f"moments and dos time: {t_moments_dos_end-t_moments_dos_start}")