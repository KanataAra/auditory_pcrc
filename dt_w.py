#coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
# Dynamic Time Warping Library
# pip install dtwでインストール可能
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

d_a=np.zeros(300)
d_b=np.zeros(300)
d_c=np.zeros(300)

for i in range(300):
    d_a[i]=np.sin(i/10)
    d_b[i]=np.sin((i-50)/10)
    d_c[i]=0

mse_1=mean_squared_error(d_a,d_b)
mse_2=mean_squared_error(d_a,d_c)
distance1, path = fastdtw(d_a, d_b, dist=euclidean)
distance2, path = fastdtw(d_a, d_c, dist=euclidean)
print("The MSE between A and B : %f" % mse_1)
print("The MSE between A and C : %f" % mse_2)
print("The DTW between A and B : %f" % distance1)
print("The DTW between A and C : %f" % distance2)

    
plt.subplot(3,1,1)
plt.title("A")
plt.plot(d_a)
plt.subplot(3,1,2)
plt.title("B")
plt.plot(d_b)
plt.subplot(3,1,3)
plt.title("C")
plt.plot(d_c)
plt.show()
