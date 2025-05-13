import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats

#this script loads out_dos.csv and plots
if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
Nm=int(sys.argv[2])
R=int(sys.argv[3])

csvDataFolderRoot=f"../dataAll/N{N}/Nm{Nm}/R{R}/"

inCsvFile=csvDataFolderRoot+"/out_dos_parallel.csv"
df=pd.read_csv(inCsvFile)
E_tilde_vec=np.array(df["E_tilde"])
dos_vec=np.array(df["dos"])
plt.figure()
plt.plot(E_tilde_vec,dos_vec/np.max(dos_vec),color="blue")
plt.title(f"N={N}, Nm={Nm}, R={R}")
plt.xlabel(r"$E$"+" (eV)")
plt.ylabel(r"DOS")
plt.savefig(csvDataFolderRoot+"/dos.png")
plt.close()


