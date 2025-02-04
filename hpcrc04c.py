#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# UNDONE: PCRCの基本的なコードを実装する。クラスは使わずに。
# Sample Run Code
# $ python hpcrc04.py id=0 seed=0 r1=0.5 r2=0.4 a=0 l1=0
# ex0
# 0.5
# 1.23189986997
# 0,0,0.500000,0.400000,0.000000,0.500000 1.231900 17.025024


import numpy as np
import sys
import csv
import scipy.linalg
import re
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
from pylab import *
from sklearn.decomposition import NMF


args=sys.argv
T1 = 2764
T0 = 100

Nx1 = 100
Ny1 = 256
Nx2 = 100
Ny2 = 128

#sigma_np = -5
alpha_r1 = 0.7
alpha_b1 = 0.7
alpha_t1 = 0.7
beta_b1 = 0.1
beta_r1 = 0.1
beta_t1 = 0.1
alpha_r2 = 0.4
alpha_b2 = 0.7
beta_b2 = 0.1
beta_r2 = 0.1
ganma_r1=0.0
ganma_r2=0.0
alpha_nmf=0.5
l1_ratio_nmf=0.5

alpha0_1 = 0.7
tau1 = 1
kx1 = 0
kr1 = 0
alpha0_2 = 0.7
tau2 = 1
kx2 = 0
kr2 = 0

e1=0
e2=0

id=0
seed=0
ex = "ex0"

lambda0 = 0.1


def fileout():
    global Y1,D1,Y2,R1,e1,e2
    str="%d,%d,%f,%f,%f,%f %f %f\n" % (id,seed,alpha_r1,alpha_r2,alpha_nmf,l1_ratio_nmf,e1,e2)
    print(str)
    filename_tmp="data_"+ex+".csv"
    f=open(filename_tmp,"a")
    f.write(str)
    f.close()

def arg2f(x,r,s):
    # transform arguments to float value
    fl = "([+-]?[0-9]+[.]?[0-9]*)" # regular expression of float value
    m=re.findall(r+fl,s)
    if m : x=float(m[0])
    return x

def arg2i(x,r,s):
    # transform arguments to float value
    tmp = "([+-]?[0-9]*)" # regular expression of integer value
    m=re.findall(r+tmp,s)
    if m : x=int(m[0])
    return x


def generate_data_sequence():
    global T1,Ny1
    global D1
    global A1
    f=open('bach434.txt','r')
    n=0
    D1 = np.zeros((T1, Ny1))
    A = np.zeros(T1*Ny1)
    for line in f:
        A[n]=line
        n=n+1
    f.close()
    for i in range(Ny1):
        for j in range(T1):
           # print(A[j*Ny+i])
            D1[j,i]=A[j*Ny1+i]/1000000
    #print(D)

def nmf():
    global R1,D2
    k=Ny2
    model = NMF(n_components=k, init='random', random_state=0,alpha=alpha_nmf,l1_ratio=l1_ratio_nmf)
    U = model.fit_transform(R1.T)
    D2_T = model.components_
    print(np.sum(U))
    print(np.sum(D2_T))
    R_re=U@D2_T
    D2_re=U.T@R1.T
    D2=D2_T.T
    sum=0
    for i in range (Ny2):
        #            sum=np.linalg.norm(R1.T[i,:]-R_re[i,:])
        sum=np.linalg.norm(D2_T[i,:]-D2_re[i,:])
    print(sum/Ny2)
    
def generate_weight_matrix_lower():
    global Wr1, Wb1, Wo1,U
    ### Wr_in_lower
    Wr0 = np.zeros(Nx1 * Nx1)
    nonzeros = Nx1 * Nx1 * beta_r1
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.seed(seed)
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx1, Nx1))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr1 = Wr0 / lambda_max * alpha_r1
    E = np.identity(Nx1)
    Wr1 = Wr1+E*ganma_r1

    #print("lamda_max", lambda_max)
    #print("Wr:")
    #print(Wr)

    ### Wb_in_lower
    Wb1 = np.zeros(Nx1 * Ny1)
    Wb1[0:int(Nx1 * Ny1 * beta_r1 / 2)] = 1
    Wb1[int(Nx1 * Ny1 * beta_r1 / 2):int(Nx1 * Ny1 * beta_r1)] = -1
    np.random.shuffle(Wb1)
    Wb1 = Wb1.reshape((Nx1, Ny1))
    Wb1 = Wb1 * alpha_b1
    #print("Wb:")
    #print(Wb)
    
    ### Wo_in_lower
    Wo1 = np.ones(Ny1 * Nx1)
    Wo1 = Wo1.reshape((Ny1, Nx1))
    Wo1 = Wo1

    ### Wt

    U = np.zeros((Ny1,Ny2))
    
    
def generate_weight_matrix_higher():
    global Wr2, Wb2, Wo2
    ### Wr_in_higher
    Wr0_2 = np.zeros(Nx2 * Nx2)
    nonzeros_2 = Nx2 * Nx2 * beta_r2
    Wr0_2[0:int(nonzeros_2 / 2)] = 1
    Wr0_2[int(nonzeros_2 / 2):int(nonzeros_2)] = -1
    np.random.seed(seed+1)
    np.random.shuffle(Wr0_2)
    Wr0_2 = Wr0_2.reshape((Nx2, Nx2))
    v_2 = scipy.linalg.eigvals(Wr0_2)
    lambda_max_2 = max(abs(v_2))
    Wr2 = Wr0_2 / lambda_max_2 * alpha_r2
    E = np.identity(Nx2)
    Wr2 = Wr2+E*ganma_r2

    ### Wb_in_higher
    Wb2 = np.zeros(Nx2 * Ny2)
    Wb2[0:int(Nx2 * Ny2 * beta_r2 / 2)] = 1
    Wb2[int(Nx2 * Ny2 * beta_r2 / 2):int(Nx2 * Ny2 * beta_r2)] = -1
    np.random.shuffle(Wb2)
    Wb2 = Wb2.reshape((Nx2, Ny2))
    Wb2 = Wb2 * alpha_b2
    #print("Wb:")
    #print(Wb)

    
    ### Wo_in_higher
    Wo2 = np.ones(Ny2*Nx2)
    Wo2 = Wo2.reshape((Ny2, Nx2))
    Wo2 = Wo2
   # print(Wo)


def config():
    global id,seed,alpha_r1,alpha_r2,alpha_nmf,l1_ratio_nmf
    args = sys.argv
    for s in args:
        id = arg2i(id,"id=",s)
        seed = arg2i(seed,"seed=",s)
        alpha_r1 = arg2f(alpha_r1,"r1=",s)
        alpha_r2 = arg2f(alpha_r2,"r2=",s)
        alpha_nmf = arg2f(alpha_nmf,"a=",s)
        l1_ratio_nmf = arg2f(l1_ratio_nmf,"l1=",s)
    print(ex)
   
def fx(x):
    return np.tanh(x)


def fy(x):
    return np.tanh(x)


def fyi(x):
    return np.arctanh(x)


def fr(x):
    return np.fmax(0, x)


def run_network_lower(mode):
    global X1, Y1, R1, D1, Wo1
    X1 = np.zeros((T1, Nx1))
    Y1 = np.zeros((T1, Ny1))
    R1 = np.zeros((T1, Ny1))
    n = 0
    x1 = np.random.uniform(-1, 1, Nx1) * 0.2
    y1 = np.zeros(Ny1)
    d1 = np.zeros(Ny1)
    r1 = np.zeros(Ny1)
    X1[n, :] = x1
    Y1[n, :] = y1
    R1[n, :] = r1
    for n in range(T1 - 1):
        xd1 = X1[n, :]
        if mode==0:
            yd1 = D1[n, :]
        else:
            yd1 = Y1[n, :]
        yr1 = R1[n, :]

        sum = np.zeros(Nx1)
        sum += Wr1@xd1
        sum += Wb1@yd1
        sum += Wb1@yr1
        x1 = x1 + 1.0 / tau1 * (-alpha0_1 * x1 + fx(sum))
        y1 = fy(Wo1@x1)
        d1 = D1[n, :]
        r1 = fr(d1 - y1)
        #        sum2 = np.zeros(Ny)
        #        sum2 = sum2 + r
        """for j in range(Ny1):
            if r1[j]<0.2:
                r1[j]=r1[j]
            if r1[i]>0.8:
                r1[i]=0.8"""
        r1 = fr(r1)
      
#        print(r)
#        r=sum2
#        sum2 = 0
        X1[n + 1, :] = x1
        Y1[n + 1, :] = y1
        R1[n + 1, :] = r1
        #print(y)

    
def run_network_higher(mode):
    global X1, Y1, R1, X2, Y2, R2, D2 ,Wo2
    X2 = np.zeros((T1, Nx2))
    Y2 = np.zeros((T1, Ny2))
    R2 = np.zeros((T1, Ny2))
    n = 0
    x2 = np.random.uniform(-1, 1, Nx2) * 0.2
    y2 = np.zeros(Ny2)
    d2 = np.zeros(Ny2)
    r2 = np.zeros(Ny2)
    X2[n, :] = x2
    Y2[n, :] = y2
    R2[n, :] = r2
    D2=R1
    for n in range(T1 - 1):
        xd2 = X2[n, :]
        if mode==0:
            yd2 = D2[n, :]
        else:
            yd2 = Y2[n, :]
        yr2 = R2[n, :]
        sum = np.zeros(Nx)
        sum += Wr2@xd2
        sum += Wb2@yd2
        sum += Wb2@yr2
        x2 = x2 + 1.0 / tau2 * (-alpha0_2 * x2 + fx(sum))
        y2 = fy(Wo2@x2)
        d2 = D2[n, :]
        r2 = fr(d2 - y2)

        X2[n + 1, :] = x2
        Y2[n + 1, :] = y2
        R2[n + 1, :] = r2
        #print(y)

        
def run_network_joint(mode):
    global X1, Y1, R1, D1, X2, Y2, R2, D2, Wo1, Wo2,U
    X1 = np.zeros((T1, Nx1))
    Y1 = np.zeros((T1, Ny1))
    R1 = np.zeros((T1, Ny1))
    X2 = np.zeros((T1, Nx2))
    Y2 = np.zeros((T1, Ny2))
    R2 = np.zeros((T1, Ny2))
    D2 = np.zeros((T1, Ny2))
    n = 0
    x1 = np.random.uniform(-1, 1, Nx1) * 0.2
    x2 = np.random.uniform(-1, 1, Nx2) * 0.2
    y1 = np.zeros(Ny1)
    y2 = np.zeros(Ny2)
    d1 = np.zeros(Ny1)
    d2 = np.zeros(Ny2)
    r1 = np.zeros(Ny1)
    r2 = np.zeros(Ny2)
    X1[n, :] = x1
    Y1[n, :] = y1
    R1[n, :] = r1
    X2[n, :] = x2
    Y2[n, :] = y2
    R2[n, :] = r2
    for n in range(T1 - 1):
        xd1 = X1[n, :]
        if mode==0:
            yd1 = D1[n, :]
        else:
            yd1 = Y1[n, :]
        y2 = np.copy(Y2[n, :])
        yr1 = U@y2
        sum1 = np.zeros(Nx1)
        sum1 += Wr1@xd1
        sum1 += Wb1@yd1
        sum1 += Wb1@yr1
        x1 = x1 + 1.0 / tau1 * (-alpha0_1 * x1 + fx(sum1))
        y1 = fy(Wo1@x1)
        d1 = D1[n+1, :]
      #  print(y)
        r1 = fr(d1 - y1)
        
        X1[n + 1, :] = x1
        Y1[n + 1, :] = y1
        R1[n + 1, :] = r1
     #   D2=np.copy(R)
        xd2 = X2[n, :]
        D2[n,:]=U.T@R1[n,:]
        if mode==0:
            yd2 = np.copy(R1[n, :])
            yd2 = U.T@yd2
        else:
            yd2 = Y2[n, :]
        yr2 = R2[n, :]
        
        sum2 = np.zeros(Nx2)
        sum2 += Wr2@xd2
        sum2 += Wb2@yd2
        sum2 += Wb2@yr2
        x2 = x2 + 1.0 / tau2 * (-alpha0_2 * x2 + fx(sum2))
        y2 = fy(Wo2@x2)
#        print(y2)
        r1 = R1[n, :]
        d2 = U.T@r1
#        print(d2)
        r2 = fr(d2 - y2)

        X2[n + 1, :] = x2
        Y2[n + 1, :] = y2
        R2[n + 1, :] = r2
#        D2[n + 1, :] = d2

def train_network_lower():
    global Wo1,U,D2,R1

    run_network_lower(0)

    M1 = X1[T0:, :]
    invD = fyi(D1)
    G1 = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx1)
    TMP1 = inv(M1.T@M1 + lambda0 * E)
    WoT1 = TMP1@M1.T@G1
    Wo1 = WoT1.T
    # print(Wo)

   
    run_network_lower(0)
       
   
def train_network_higher():
    global Wo2, D2, R

    run_network_higher(0)

    M2 = X2[T0:, :]
 #   D2 = R
    invD2 = fyi(D2)
    G2 = invD2[T0:, :]

    ### Ridge regression
    E = np.identity(Nx2)
    TMP1_2 = inv(M2.T@M2 + lambda0 * E)
    WoT_2 = TMP1_2@M2.T@G2
    Wo2 = WoT_2.T
   # print(Wo)

def train_network_joint():
    global Wo1,Wo2,R1,X1,X2,D2,U,alpha_nmf,l1_ratio_nmf

    run_network_joint(0)
    M1 = X1[T0:, :]
    invD = fyi(D1)
    G1 = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx1)
    TMP1 = inv(M1.T@M1 + lambda0 * E)
    WoT1 = TMP1@M1.T@G1
    Wo1 = WoT1.T
    # print(Wo)

        
    #    run_network_joint(0)
    M2 = X2[T0:, :]
    invD2 = fyi(D2)
    G2 = invD2[T0:, :]
    
    ### Ridge regression
    E2 = np.identity(Nx2)
    TMP1_2 = inv(M2.T@M2 + lambda0 * E)
    WoT2 = TMP1_2@M2.T@G2
    Wo2 = WoT2.T
    # print(Wo)"

        
def analyze_error():
    global e1,Y1,D1,e2,Y2,D2
    sum1=0
    sum2=0
    for i in range(T1):
        sum1+=np.linalg.norm(Y1[i,:]-D1[i,:])
        sum2+=np.linalg.norm(Y2[i,:]-D2[i,:])
    e1=sum1/T1
    e2=sum2/T1

def plot():
    fig=plt.figure()
    ax1 = fig.add_subplot(4,2,1)
    ax1.cla()
    ax1.set_title("R1")
    ax1.plot(R1)
    ax2 = fig.add_subplot(4,2,3)
    ax2.cla()
    ax2.set_title("D1")
    ax2.plot(D1)
    ax3 = fig.add_subplot(4,2,5)
    ax3.cla()
    ax3.set_title("Y1")
    ax3.plot(Y1)
    ax4 = fig.add_subplot(4,2,7)
    ax4.cla()
    ax4.set_title("X1")
    ax4.plot(X1)
    ax5 = fig.add_subplot(4,2,2)
    ax5.cla()
    ax5.set_title("R2")
    ax5.plot(R2)
    ax6 = fig.add_subplot(4,2,4)
    ax6.cla()
    ax6.set_title("D2")
    ax6.plot(D2)
    ax7 = fig.add_subplot(4,2,6)
    ax7.cla()
    ax7.set_title("Y2")
    ax7.plot(Y2)
    ax8 = fig.add_subplot(4,2,8)
    ax8.cla()
    ax8.set_title("X2")
    ax8.plot(X2)
    plt.show()

def plot_nmf():    

    x=arange(T1)
    y=arange(Ny1)
    
    X,Y=meshgrid(x,y)
    #        pcolor(X, Y, np.log(R.T+0.0001))
    pcolor(X, Y, R1.T)
    colorbar
    show()

    K=arange(k)
    X,KK=meshgrid(x,K)
    #        pcolor(X, KK, np.log(D2+0.0001))
    pcolor(X, KK, D2)
    colorbar
    show()
    
    KK,Y=meshgrid(K,y)
    #        pcolor(KK, y, np.log(U+0.0001))
    pcolor(KK, y, U)
    colorbar
    show()

    
def Execute():
    generate_weight_matrix_lower()
    generate_data_sequence()
    generate_weight_matrix_higher()
    
    train_network_lower()
    nmf()
    #    for m in range(10):
    for n in range(5):
        train_network_joint()
#        nmf()
    run_network_joint(1)
    analyze_error()

if __name__ == "__main__":

    config()
    Execute()
    fileout()
    plot()

#    with open('new_training2.csv', 'a') as f:
#        writer = csv.writer(f)
#        writer.writerow([alpha_r,check_error()])
