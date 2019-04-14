#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# UNDONE: PCRCの基本的なコードを実装する。クラスは使わずに。

import numpy as np
import sys
import csv
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

args=sys.argv
T1 = 1838
T0 = 5

Nx = 100
Ny = 256
Nx2 = 100
Ny2 = 256

#sigma_np = -5
alpha_r = float(args[1])
alpha_b = 0.7
beta_b = 0.1
beta_r = 0.1
alpha_r2 = 0.4
alpha_b2 = 0.7
beta_b2 = 0.1
beta_r2 = 0.1

alpha0 = 0.7
tau = 1
kx = 0
kr = 0
alpha0_2 = 0.7
tau2 = 1
kx2 = 5
kr2 = 0

lambda0 = 0.1


def generate_data_sequence():
    global T1,Ny
    global D
    global A
    f=open('phrase1.txt','r')
    n=0
    D = np.zeros((T1, Ny))
    A = np.zeros(T1*Ny)
    for line in f:
        A[n]=line
        n=n+1
    f.close()
    for i in range(Ny):
        for j in range(T1):
           # print(A[j*Ny+i])
            D[j,i]=A[j*Ny+i]/500000
    print(A)

def generate_weight_matrix():
    global Wr, Wb, Wo,Wr2, Wb2, Wo2
    ### Wr_in_lower
    Wr0 = np.zeros(Nx * Nx)
    nonzeros = Nx * Nx * beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx, Nx))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr = Wr0 / lambda_max * alpha_r


    ### Wr_in_higher
    Wr0_2 = np.zeros(Nx * Nx)
    nonzeros_2 = Nx * Nx * beta_r2
    Wr0_2[0:int(nonzeros_2 / 2)] = 1
    Wr0_2[int(nonzeros_2 / 2):int(nonzeros_2)] = -1
    np.random.shuffle(Wr0_2)
    Wr0_2 = Wr0_2.reshape((Nx, Nx))
    v_2 = scipy.linalg.eigvals(Wr0_2)
    lambda_max_2 = max(abs(v_2))
    Wr2 = Wr0_2 / lambda_max_2 * alpha_r
    
    #print("lamda_max", lambda_max)
    #print("Wr:")
    #print(Wr)

    ### Wb_in_lower
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_r / 2)] = 1
    Wb[int(Nx * Ny * beta_r / 2):int(Nx * Ny * beta_r)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb = Wb * alpha_b
    #print("Wb:")
    #print(Wb)


    ### Wb_in_higher
    Wb2 = np.zeros(Nx * Ny)
    Wb2[0:int(Nx * Ny * beta_r2 / 2)] = 1
    Wb2[int(Nx * Ny * beta_r2 / 2):int(Nx * Ny * beta_r2)] = -1
    np.random.shuffle(Wb2)
    Wb2 = Wb2.reshape((Nx, Ny))
    Wb2 = Wb2 * alpha_b
    #print("Wb:")
    #print(Wb)
    
    ### Wo_in_lower
    Wo = np.ones(Ny * Nx)
    Wo = Wo.reshape((Ny, Nx))
    Wo = Wo
    
    ### Wo_in_higher
    Wo2 = np.ones(Ny * Nx)
    Wo2 = Wo.reshape((Ny, Nx))
    Wo2 = Wo2
   # print(Wo)


def fx(x):
    return np.tanh(x)


def fy(x):
    return np.tanh(x)


def fyi(x):
    return np.arctanh(x)


def fr(x):
    return np.fmax(0, x)


def run_network_lower(mode):
    global X, Y, R, D, Wo
    X = np.zeros((T1, Nx))
    Y = np.zeros((T1, Ny))
    R = np.zeros((T1, Ny))
    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.2
    y = np.zeros(Ny)
    d = np.zeros(Ny)
    r = np.zeros(Ny)
    X[n, :] = x
    Y[n, :] = y
    R[n, :] = r
    for n in range(T1 - 1):
        xd = X[n, :]
        if mode==0:
            yd = D[n, :]
        else:
            yd = Y[n, :]
        yr = R[n, :]

        sum = np.zeros(Nx)
        sum += Wr@xd
        sum += Wb@yd
        sum += Wb@yr
        x = x + 1.0 / tau * (-alpha0 * x + fx(sum))
        y = fy(Wo@x)
        d = D[n, :]
        r = fr(d - y)
        #        sum2 = np.zeros(Ny)
        #        sum2 = sum2 + r
        for j in range(Ny):
            if r[j]<0.2:
                r[j]=r[j]
 #       if n<10:
 #           for i in range(n):
 #               r=r+R[n-(i+1),:]*0.5**(i+1)
 #       else:
 #           for i in range(10):
 #               r=r+R[n-(i+1),:]*0.5**(i+1)
 #       
        for i in range(Ny):
            if r[i]>0.8:
                r[i]=0.8
        r = fr(r)
      
#        print(r)
#        r=sum2
#        sum2 = 0
        X[n + 1, :] = x
        Y[n + 1, :] = y
        R[n + 1, :] = r
        #print(y)

    
def run_network_higher(mode):
    global X, Y, R, X2, Y2, R2, D2 ,Wo2
    X2 = np.zeros((T1, Nx))
    Y2 = np.zeros((T1, Ny))
    R2 = np.zeros((T1, Ny))
    n = 0
    x2 = np.random.uniform(-1, 1, Nx2) * 0.2
    y2 = np.zeros(Ny)
    d2 = np.zeros(Ny)
    r2 = np.zeros(Ny)
    X2[n, :] = x2
    Y2[n, :] = y2
    R2[n, :] = r2
    D2=R
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
    global X, Y, R, D, X2, Y2, R2, D2, Wo, Wo2
    X = np.zeros((T1, Nx2))
    Y = np.zeros((T1, Ny2))
    R = np.zeros((T1, Ny2))
    X2 = np.zeros((T1, Nx2))
    Y2 = np.zeros((T1, Ny2))
    R2 = np.zeros((T1, Ny2))
    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.2
    x2 = np.random.uniform(-1, 1, Nx2) * 0.2
    y = np.zeros(Ny)
    y2 = np.zeros(Ny2)
    d = np.zeros(Ny)
    d2 = np.zeros(Ny)
    r = np.zeros(Ny)
    r2 = np.zeros(Ny2)
    X[n, :] = x
    Y[n, :] = y
    R[n, :] = r
    X2[n, :] = x2
    Y2[n, :] = y2
    R2[n, :] = r2
    for n in range(T1 - 1):
        xd = X[n, :]
        if mode==0:
            yd = D[n, :]
        else:
            yd = Y[n, :]
        if n==0:
            yr = R[n, :]
        else:
            yr = Y2[n, :]
        sum = np.zeros(Nx)
        sum += Wr@xd
#        sum += Wb@yd
        sum += Wb@yr
        x = x + 1.0 / tau * (-alpha0 * x + fx(sum))
        y = fy(Wo@x)
        d = D[n, :]
        r = fr(d - y)

        X[n + 1, :] = x
        Y[n + 1, :] = y
        R[n + 1, :] = r
        D2=R
        xd2 = X2[n, :]
        if mode==0:
            yd2 = R[n, :]
        else:
            yd2 = Y2[n, :]
        yr2 = R2[n, :]

        sum2 = np.zeros(Nx2)
        sum2 += Wr2@xd2
        sum2 += Wb2@yd2
        sum2 += Wb2@yr2
        x2 = x2 + 1.0 / tau2 * (-alpha0_2 * x2 + fx(sum2))
        y2 = fy(Wo2@x2)
        d2 = R[n, :]
        r2 = fr(d2 - y2)

        X2[n + 1, :] = x2
        Y2[n + 1, :] = y2
        R2[n + 1, :] = r2
        

def train_network_lower():
    global Wo

    run_network_lower(0)


    M = X[T0:, :]
    invD = fyi(D)
    G = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T
   # print(Wo)

   
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


def check_error():
    global Y,D
    sum=0
    for i in range(T1):
        sum+=np.linalg.norm(Y[i,:]-D[i,:])
    return sum/T1

def plot():
    fig=plt.figure()
    ax1 = fig.add_subplot(6,1,1)
    ax1.cla()
    ax1.plot(R)
    ax2 = fig.add_subplot(6,1,2)
    ax2.cla()
    ax2.plot(D)
    ax3 = fig.add_subplot(6,1,3)
    ax3.cla()
    ax3.plot(Y)
    ax4 = fig.add_subplot(6,1,4)
    ax4.cla()
    ax4.plot(R2)
    ax5 = fig.add_subplot(6,1,5)
    ax5.cla()
    ax5.plot(D2)
    ax6 = fig.add_subplot(6,1,6)
    ax6.cla()
    ax6.plot(Y2)
    plt.show()

    
def Execute():
    generate_weight_matrix()
    generate_data_sequence()
    for n in range(10):
        train_network_lower()
    run_network_lower(1)
    for n in range(10):
        train_network_higher()
    run_network_higher(1)
    run_network_joint(1)
    
if __name__ == "__main__":
    Execute()
    print(alpha_r)
    print(check_error())
    plot()
#    with open('error01.csv', 'a') as f:
#        writer = csv.writer(f)
#        writer.writerow([alpha_r,check_error()])

#    run_network_joint()
