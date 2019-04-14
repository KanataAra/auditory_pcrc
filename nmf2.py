from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
T1=1838
Ny=256
R = np.zeros((Ny, T1))


f=open('phrase1.txt','r')
n=0
A = np.zeros(T1*Ny)
for line in f:
    A[n]=line
    n=n+1
f.close()
for i in range(Ny):
    for j in range(T1):
        R[i,j]=A[j*Ny+i]/500000

k=16
model = NMF(n_components=k, init='random', random_state=0)
W = model.fit_transform(R)
H = model.components_
print(H)


fig=plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.cla()
ax1.imshow(R)

ax2 = fig.add_subplot(2,1,2)
ax2.cla()
ax2.imshow(H)


plt.show()
