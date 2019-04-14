from sklearn.decomposition import NMF
from pylab import *
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
T1=1720
#10335

Ny=256
R = np.zeros((Ny, T1))


f=open('chord_steeply.txt','r')
n=0
A = np.zeros(T1*Ny)
for line in f:
    A[n]=line
    n=n+1
f.close()
for i in range(Ny):
    for j in range(T1):
        R[i,j]=A[j*Ny+i]/50000

k=64
model = NMF(n_components=k, init='random', random_state=0)
P = model.fit_transform(R)
Q = model.components_
print(Q)


#fig=plt.figure()
#ax1 = fig.add_subplot(3,1,1)
#ax1.cla()
#ax1.imshow(np.log(R+0.0001))
#ax1.imshow(R)

#plt.imshow(np.log(R+0.0001))

x=arange(T1+1)
y=arange(Ny+1)

X,Y=meshgrid(x,y)
pcolor(X, Y, np.log(R+0.0001))
colorbar
show()

K=arange(k)
X,KK=meshgrid(x,K)
pcolor(X, KK, np.log(Q+0.0001))
colorbar
show()

KK,Y=meshgrid(K,y)
pcolor(KK, y, np.log(P+0.0001))
colorbar
show()
"""
ax2 = fig.add_subplot(3,1,2)
ax2.cla()
ax2.imshow(np.log(Q+0.0001))
ax2.imshow(Q)

plt.savefig("moonlight.eps")
"""
#fig2=plt.figure()
#ax3 = fig.add_subplot(3,1,3)
#ax3.cla()
#ax3.imshow(np.log(P+0.0001))
#ax3.imshow(P)

#plt.savefig("moonlight2.eps")

#plt.show()

#fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(10,4)
#ax1.imshow(np.log(R+0.0001))
#ax2.imshow(np.log(Q+0.0001))

