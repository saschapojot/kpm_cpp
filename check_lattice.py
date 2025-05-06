import numpy as np
import matplotlib.pyplot as plt

a1=[1/2,
    3**.5/2]
a2=[-1/2, 3**.5/2]
delta=1/3**.5

Rx_list=[]
Ry_list=[]

N1=3
N2=3

fig,ax=plt.subplots()
ax.set_aspect("equal")

for i1 in range(N1):
    for i2 in range(N2):
        Rx=i1*a1[0]+i2*a2[0]
        Ry=i1*a1[1]+i2*a2[1]
        Rx_list.append(Rx)
        Ry_list.append(Ry)
        ind_str1="("+str(i1)+", "+str(i2)+")" # string for the indices
        ind_str2=str(i1*N2+i2) # string for the indices
        ax.text(Rx,Ry,ind_str1)
        ax.text(Rx,Ry+0.25,ind_str2)
        
Rx_list=np.array(Rx_list)
Ry_list=np.array(Ry_list)
ax.scatter(Rx_list,Ry_list)
ax.scatter(Rx_list,Ry_list+delta)

def get_neighbors(ind,sublattice):
    global N1,N2
    #ind=m1*N2+n1
    m1=ind//N2
    n1=ind%N2
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

print(get_neighbors(2,"A"))
print(get_neighbors(2,"B"))


plt.savefig("lattice.png")
















