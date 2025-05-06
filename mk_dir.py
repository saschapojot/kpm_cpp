from pathlib import Path
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
#########this script initializes directories and cppIn.txt files
def format_using_decimal(value, precision=5):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


# N=20 #unit cell number
# R=50 # number of random vectors
parallel_num=8 # number of jobs in parallel
# Nm=500 # number of moments
N_moments_vec=[20,100,500] # number of moments
N_vec=[10,20]#unit cell number
R_vec=[20,50]# number of random vectors
lamb=1  #not used if it is not lorentz
t0=2.7
dataOutDir="./dataAll/"

def contents_to_conf(ind_N,ind_Nm,ind_R):
    """

    :param ind_N: index in N_vec
    :param ind_Nm: index in N_moments_vec
    :param ind_R: index in R_vec
    :return:
    """
    Nm_Str=format_using_decimal(N_moments_vec[ind_Nm])
    N_Str=format_using_decimal(N_vec[ind_N])
    R_Str=format_using_decimal(R_vec[ind_R])
    lamb_Str=format_using_decimal(lamb)
    t0_Str=format_using_decimal(t0)
    parallel_num_Str=format_using_decimal(parallel_num)

    contents=[
        f"{N_Str}\n",
        f"{Nm_Str}\n",
        f"{R_Str}\n",
        f"{lamb_Str}\n",
        f"{t0_Str}\n",
        f"{parallel_num_Str}\n"
        ]
    outDir=dataOutDir+f"/N{N_Str}/Nm{Nm_Str}/R{R_Str}/"
    Path(outDir).mkdir(exist_ok=True,parents=True)
    outConfName=outDir+"/cppIn.txt"
    with open(outConfName,"w+")as fptr:
        fptr.writelines(contents)


for i in range(0,len(N_vec)):
    for j in range(0,len(N_moments_vec)):
        for k in range(0,len(R_vec)):
            contents_to_conf(i,j,k)