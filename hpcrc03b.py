#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Katori Lab. All Rights Reserved
# NOTE: hierarchical PCRC

from time import sleep
import numpy as np
import sys
import re
import csv
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

args=sys.argv
T1 = 1838
T0 = 100

Nx1 = 100
Ny1 = 256
Nx2 = 100
Ny2 = 256

#sigma_np = -5
alpha_r1 = 0.7
alpha_b1 = 0.7
beta_b1 = 0.1
beta_r1 = 0.1
gamma_r1=0.0
alpha0_1 = 0.7
tau1 = 1
kx1 = 0
kr1 = 0

alpha_r2 = 0.4 #0.4
alpha_b2 = 0.7
beta_b2 = 0.1
beta_r2 = 0.1
gamma_r2=0.35 #0.4
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
    #global Y,D,Y2,R,e1,e2
    str="%d,%d,%f,%f,%f,%f,%f,%f,%f,%f\n" % (id,seed,alpha_r1,alpha_b1,gamma_r1,alpha_r2,alpha_b2,gamma_r2,e1,e2)
    print(str)
    filename_tmp="data_"+ex+".csv"
    f=open(filename_tmp,"a")
    f.write(str)
    f.close()

def config():
    global ex,id,seed,alpha_r1,alpha_b1,gamma_r1,alpha_r2,alpha_b2,gamma_r2
    args = sys.argv
    for s in args:
        ex = arg2a(ex,"ex=",s)
        id = arg2i(id,"id=",s)
        seed = arg2i(seed,"seed=",s)
        alpha_r1 = arg2f(alpha_r1,"alpha_r1=",s)
        alpha_b1 = arg2f(alpha_b1,"alpha_b1=",s)
        gamma_r1 = arg2f(gamma_r1,"gamma_r1=",s)
        alpha_r2 = arg2f(alpha_r2,"alpha_r2=",s)
        alpha_b2 = arg2f(alpha_b2,"alpha_b2=",s)
        gamma_r2 = arg2f(gamma_r2,"gamma_r2=",s)

    print(ex)

def arg2f(x,r,s):
    # transform arguments to float value
    fl = "([+-]?[0-9]+[.]?[0-9]*)" # regular expression of float value
    m=re.findall(r+fl,s)
    if m : x=float(m[0])
    return x

def arg2i(x,r,s):
    # transform arguments to integer value
    tmp = "([+-]?[0-9]*)" # regular expression of integer value
    m=re.findall(r+tmp,s)
    if m : x=int(m[0])
    return x

def arg2a(x,r,s):
    tmp = "(\w*)" # regular expression of a-z,A-Z,0-9,and _
    m=re.findall(r+tmp,s)
    if m : x=m[0]
    return x

def generate_data_sequence():
    global T1,Ny1
    global D
    global A
    f=open('phrase1.txt','r')
    n=0
    D = np.zeros((T1, Ny1))
    A = np.zeros(T1*Ny1)
    for line in f:
        A[n]=line
        n=n+1
    f.close()
    for i in range(Ny1):
        for j in range(T1):
           # print(A[j*Ny+i])
            D[j,i]=A[j*Ny1+i]/500000

    #print(D)

def generate_weight_matrix_lower():
    global Wr1, Wb1, Wo1
    ### Wr_in_lower
    Wr0 = np.zeros(Nx1 * Nx1)
    nonzeros = Nx1 * Nx1 * beta_r1
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    #np.random.seed(seed)
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx1, Nx1))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr1 = Wr0 / lambda_max * alpha_r1
    E = np.identity(Nx1)
    Wr1 = Wr1 + gamma_r1*E

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

def generate_weight_matrix_higher():
    global Wr2, Wb2, Wo2
    ### Wr_in_higher
    Wr0 = np.zeros(Nx2 * Nx2)
    nonzeros = Nx2 * Nx2 * beta_r2
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    #np.random.seed(seed)
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx2, Nx2))
    v_2 = scipy.linalg.eigvals(Wr0)
    lambda_max_2 = max(abs(v_2))
    Wr2 = Wr0 / lambda_max_2 * alpha_r2
    E = np.identity(Nx2)
    Wr2 = Wr2 + gamma_r2*E

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
    Wo2 = np.ones(Ny2 * Nx2)
    Wo2 = Wo2.reshape((Ny2, Nx2))
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
    global X1, Y1, R1, D, Wo1
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
            yd1 = D[n, :]
        else:
            yd1 = Y1[n, :]
        yr1 = R1[n, :]

        sum = np.zeros(Nx1)
        sum += Wr1@xd1
        sum += Wb1@yd1
        sum += Wb1@yr1
        x1 = x1 + 1.0 / tau1 * (-alpha0_1 * x1 + fx(sum))
        y1 = fy(Wo1@x1)
        d1 = D[n, :]
        r1 = fr(d1 - y1)
        #        sum2 = np.zeros(Ny)
        #        sum2 = sum2 + r
        for j in range(Ny1):
            if r1[j]<0.2:
                r1[j]=r1[j]
 #       if n<10:
 #           for i in range(n):
 #               r=r+R[n-(i+1),:]*0.5**(i+1)
 #       else:
 #           for i in range(10):
 #               r=r+R[n-(i+1),:]*0.5**(i+1)
 #
        for i in range(Ny1):
            if r1[i]>0.8:
                r1[i]=0.8
        r1 = fr(r1)

#        print(r)
#        r=sum2
#        sum2 = 0
        X1[n + 1, :] = x1
        Y1[n + 1, :] = y1
        R1[n + 1, :] = r1
        #print(y)


def run_network_higher(mode):
    global X2, Y2, R2, D2 ,Wo2
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
        sum = np.zeros(Nx2)
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
    global X1, Y1, R1, D, X2, Y2, R2, D2, Wo, Wo2
    X1 = np.zeros((T1, Nx1))
    Y1 = np.zeros((T1, Ny1))
    R1 = np.zeros((T1, Ny1))
    X2 = np.zeros((T1, Nx2))
    Y2 = np.zeros((T1, Ny2))
    R2 = np.zeros((T1, Ny2))
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
            yd1 = D[n, :]
        else:
            yd1 = Y1[n, :]

        if n==0:
            yr1 = R1[n, :]
        else:
            yr1 = Y2[n, :]

        yr1 = Y2[n, :]
        sum = np.zeros(Nx1)
        sum += Wr1@xd1
        sum += Wb1@yd1
        sum += Wb1@yr1
        x1 = x1 + 1.0 / tau1 * (-alpha0_1 * x1 + fx(sum))
        y1 = fy(Wo1@x1)
        d1 = D[n + 1, :] # XXX
        r1 = fr(d1 - y1)

        X1[n + 1, :] = x1
        Y1[n + 1, :] = y1
        R1[n + 1, :] = r1

     #   D2=np.copy(R)
        xd2 = X2[n, :]
        if mode==0:
            yd2 = R1[n, :]
        else:
            yd2 = Y2[n, :]

        yr2 = R2[n, :]

        sum2 = np.zeros(Nx2)
        sum2 += Wr2@xd2
        sum2 += Wb2@yd2
        sum2 += Wb2@yr2
        x2 = x2 + 1.0 / tau2 * (-alpha0_2 * x2 + fx(sum2))
        y2 = fy(Wo2@x2)
        d2 = R1[n + 0, :] # XXX
        r2 = fr(d2 - y2)

        X2[n + 1, :] = x2
        Y2[n + 1, :] = y2
        R2[n + 1, :] = r2


def train_network_lower():
    global Wo1
    run_network_lower(0)

    M = X1[T0:, :]
    invD = fyi(D)
    G = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx1)
    TMP = inv(M.T@M + lambda0 * E)
    WoT = TMP@M.T@G
    Wo1 = WoT.T
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
    TMP = inv(M2.T@M2 + lambda0 * E)
    WoT = TMP@M2.T@G2
    Wo2 = WoT.T
   # print(Wo)

def train_network_joint():
    global Wo1,Wo2,R,X1,X2

    run_network_joint(0)
    M1 = X1[T0:, :]
    invD = fyi(D)
    G1 = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx1)
    TMP = inv(M1.T@M1 + lambda0 * E)
    WoT1 = TMP@M1.T@G1
    Wo1 = WoT1.T
    # print(Wo)

    M2 = X2[T0:, :]
    invD2 = fyi(R1)
    G2 = invD2[T0:, :]

    ### Ridge regression
    E2 = np.identity(Nx2)
    TMP1_2 = inv(M2.T@M2 + lambda0 * E)
    WoT2 = TMP1_2@M2.T@G2
    Wo2 = WoT2.T
    # print(Wo)

def analyze_error():
    global e1,Y1,D,e2,Y2,R1
    sum1=0
    sum2=0
    for i in range(T1):
        sum1+=np.linalg.norm(Y1[i,:]-D[i,:])
        sum2+=np.linalg.norm(Y2[i,:]-R1[i,:])
    e1=sum1/T1
    e2=sum2/T1

def plot():
    fig=plt.figure()
    ax1 = fig.add_subplot(8,1,1)
    ax1.cla()
    ax1.plot(X1)
    ax2 = fig.add_subplot(8,1,2)
    ax2.cla()
    ax2.plot(Y1)
    ax3 = fig.add_subplot(8,1,3)
    ax3.cla()
    ax3.plot(D)
    ax4 = fig.add_subplot(8,1,4)
    ax4.cla()
    ax4.plot(R1)
    ax5 = fig.add_subplot(8,1,5)
    ax5.cla()
    ax5.plot(X2)
    ax6 = fig.add_subplot(8,1,6)
    ax6.cla()
    ax6.plot(Y2)
    ax7 = fig.add_subplot(8,1,7)
    ax7.cla()
    ax7.plot(R1)
    ax8 = fig.add_subplot(8,1,8)
    ax8.cla()
    ax8.plot(R2)
    plt.show()


def Execute():
    np.random.seed(seed)
    generate_data_sequence()
    generate_weight_matrix_lower()
    generate_weight_matrix_higher()

    #train_network_lower()
    #train_network_higher()
    for n in range(10):
        train_network_joint()
    run_network_joint(0)
    analyze_error()

if __name__ == "__main__":
    config()
    Execute()
    fileout()
    plot()

#Test_pull_request#

#    with open('new_training2.csv', 'a') as f:
#        writer = csv.writer(f)
#        writer.writerow([alpha_r,check_error()])
