import numpy as np
import matplotlib.pyplot as plt
import time
import numba

# Gaussian function
def gauss(x,sigma):
    return np.exp(-x**2/2/sigma**2)
    
# Chebyshev polynomials of the first kind
def T_cheb(n,x):
    return np.cos(n*np.arccos(x))

# Simpson's rule for numerical integration
@numba.jit(nopython=True) # numba.jit speeds up the calculations
def simpson(xarray,yarray):
    # assumes len(x) is odd
    assert len(xarray)%2==1
    factor=(xarray[-1]-xarray[0])/3/(len(xarray)-1)
    intgl=0
    for i_x in range( round((len(xarray)-1)/2) ):
        i_even=round(2*i_x)
        intgl+=(yarray[i_even]+4*yarray[i_even+1]+yarray[i_even+2])
    return factor*intgl

# Kernel polynomials
def kpm(n,kernel,Nm):
    poly=0
    if kernel=="Dirichlet":
        # No kernel
        poly=1
    elif kernel=="Jackson":
        # Jackson kernel
        poly=((Nm-n+1)*np.cos(np.pi*n/(Nm+1))+np.sin(np.pi*n/(Nm+1))/np.tan(np.pi/(Nm+1)))/(Nm+1)
    elif kernel=="Lorentz":
        lamb=1
        poly=np.sinh(lamb*(1-n/Nm))/np.sinh(lamb) #lorentz
    return poly

# Expand Delta function in terms of Chebyshev moments
# Use Gaussian function with a very small sigma as Delta function
def poly_expand(sigma,Nm,x_int,x_plot,kernel):
    mu_list=np.zeros(Nm,dtype=float)
    f_int=gauss(x_int,sigma)
    f_plot=np.zeros(len(x_plot),dtype=float)
    mu0=simpson(x_int,f_int)*kpm(0,kernel,Nm)
    f_plot=mu0
    for i_mu in range(1,Nm):
        integrand=f_int*T_cheb(i_mu,x_int)
        mu_list[i_mu]=simpson(x_int,integrand) # Chebyshev moments
        f_plot+=2*mu_list[i_mu]*T_cheb(i_mu,x_plot)*kpm(i_mu,kernel,Nm)
    f_plot=1/np.pi/(1-x_plot**2)**0.5*f_plot
    return f_plot

start_time=time.time()

sigma=1e-4 # much smaller than sigma_KPM defined below
Nm=500
e_range=2 # Emax-Emin
sigma_KPM=np.pi/Nm*(e_range/2) # Eq. 76 of Rev. Mod. Phys. 78, 275 (2006)
# when Emax = -Emin, e_range/2=Emax
x_int=np.linspace(-1,1,100*Nm+1)
x_plot=np.linspace(-5*sigma_KPM,5*sigma_KPM,201)
y_original=gauss(x_plot,sigma)
y_KPM_D=poly_expand(sigma,Nm,x_int,x_plot,"Dirichlet") # without kernel polynomials
y_KPM_J=poly_expand(sigma,Nm,x_int,x_plot,"Jackson") # Jackson kernels
#%%
plt.plot(x_plot,y_original/np.max(y_original),c="C0",label=r"$\sigma=10^{-4}$",lw=2,zorder=0)
#plt.plot(x_plot,y_KPM_D/np.max(y_KPM_D),c="C1",label="Dirichlet")
plt.plot(x_plot,y_KPM_J/np.max(y_KPM_J),c="C2",label="Jackson\n"+r"$N=500$",lw=2,zorder=1)

x_scatter=np.linspace(-5*sigma_KPM,5*sigma_KPM,51)
y_scatter=gauss(x_scatter,sigma_KPM)
plt.plot(x_scatter,y_scatter,c="k",zorder=2,lw=0,marker="o",fillstyle='none')

plt.text(0.01,0.5,r"$\sigma_{KPM}\approx\frac{\pi}{N}$",fontsize=20)
plt.xlabel(r"$x$")
plt.legend()

final_time=time.time()
runtime=final_time-start_time
print("runtime",round(runtime,3),"s")



