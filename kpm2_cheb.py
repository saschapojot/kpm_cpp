import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.sparse as spa
from numba import jit
import ray
from kpm1_makeH import build_H

# Eq. 2.20-2.25 of thesis, p. 21
# see Ref. 5 of thesis, Review of Modern Physics on KPM
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
def T_cheb(n,x): # put energy as input
    return np.cos(n*np.arccos(x))

# kernel polynomial, gm, eq. 2.15 for Jackson kernel
@jit(nopython=True)
def kpm(n,Nm2,para):
    poly=((Nm2-n+1)*np.cos(np.pi*n/(Nm2+1))+np.sin(np.pi*n/(Nm2+1))/np.tan(np.pi/(Nm2+1)))/(Nm2+1) #jackson
    #poly=np.sinh(para*(1-n/Nm2))/np.sinh(para) #lorentz
    #poly=1-n/Nm2 #Feje
    return poly

# eq. 2.13 (for each random vector)
# moments are "\mu_m" in thesis
@jit(nopython=True)
def dos_kpm(e_list,Emin,Emax,moments,Nm2):
    lamb=1 #not used if it is not lorentz
    eb=(Emax+Emin)*0.5
    ea=(Emax-Emin)/(2-0.01)
    et_list=1/ea*e_list-1/ea*np.full(len(e_list),eb)#scaled energy
    d_list=np.zeros((len(et_list)))
    for iet in range(len(et_list)):
        #kpm (n,Nm2,para): values of g
        dos_recons=moments[0]*kpm(0,Nm2,lamb)+2*moments[1]*kpm(1,Nm2,lamb)*T_cheb(1,et_list[iet])
        for im in range(2,Nm2):
            mun=moments[im]
            dos_recons=dos_recons+2*mun*kpm(im,Nm2,lamb)*T_cheb(im,et_list[iet])
        d_list[iet]=1/np.sqrt(1-et_list[iet]**2)*dos_recons
    return d_list

# put two functions together and run it in parallel
@ray.remote
def calculate_moment_and_dos(e_list,H,Emin,Emax,Nm):
    moments=write_moments(H,Nm)
    d_list=dos_kpm(e_list,Emin,Emax,moments,Nm)
    return d_list

if __name__=="__main__":
    R=50 # number of random vectors
    Ncpu=8 # number of jobs in parallel
    Nm=500 # number of moments
    
    B=0.0
    N1,N2=50,50 # how many unit cells in each direction
    t1=time.time()
    H,Emin,Emax=build_H(N1,N2)
    print("Build Hamiltonian",round(time.time()-t1,5),"s")  
    
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
    ray.shutdown()
    ray.init(num_cpus=Ncpu,include_dashboard=False,log_to_driver=False)
    t2=time.time()
    e_list=np.linspace(-6,6,1001)
    d_list=np.zeros(len(e_list))
    
    ## multiprocessing
    H_ray=ray.put(H) # share it among different processes
    all_tasks=[]
    for i_R in range(R):
        all_tasks.append(calculate_moment_and_dos.remote(e_list,H_ray,Emin,Emax,Nm))
    results=ray.get(all_tasks)
    ray.shutdown()
    d_list=np.sum(results,axis=0)/R
    t4=time.time()
    print("calculate DOS",round(t4-t2,5),"s")   
    t5=time.time()
    plt.figure()
    plt.plot(e_list,d_list/np.max(d_list))
    plt.title(r"$N_1="+str(N1)+"$")
    plt.xlabel(r"$E$"+" (eV)")
    plt.ylabel(r"DOS")
    plt.xlim(-6,6)
    plt.ylim(0,1.05)
    plt.savefig("graphene_Nm"+str(Nm)+".png")
    
