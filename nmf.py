from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
T1=1838
Ny=256
R = np.zeros((T1, Ny))


f=open('phrase1.txt','r')
n=0
A = np.zeros(T1*Ny)
for line in f:
    A[n]=line
    n=n+1
f.close()
for i in range(Ny):
    for j in range(T1):
        R[j,i]=A[j*Ny+i]/500000

k=64
model = NMF(n_components=k, init='random', random_state=0)
P = model.fit_transform(R)
Q = model.components_
print(Q)


fig=plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.cla()
ax1.imshow(R.T)
ax2 = fig.add_subplot(2,1,2)
ax2.cla()
ax2.imshow(P.T)
plt.show()
