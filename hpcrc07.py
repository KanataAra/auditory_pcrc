#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)

import numpy as np
import scipy.linalg
from sklearn.metrics import mean_squared_error
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
from pylab import *
import sys
from arg2x import *
from sklearn.decomposition import NMF

#T1 = 1838
#T0 = 5

Nx1 = 100
Ny1 = 256
Nx2 = 100
Ny2 = 64

#sigma_np = -5
alpha_r1 = 0.2
alpha_b1 = 0.66
beta_r1 = 0.1
beta_b1 = 0.1
beta = 2.5
alpha1 = 0.2
tau1 = 4

#sigma_np = -5
alpha_r2 = 1.56
alpha_b2 = 0.36
beta_r2 = 0.1
beta_b2 = 0.1
#beta = 2.5
alpha2 = 0.2
tau2 = 4

alpha_nmf=0.7
l1_ratio_nmf=0.4

sigma=0
sigma1=0.000 # noise in training
sigma2=0 # noise in test

gamma=47

kx = 0
kr = 0

tauc=10
U0=0.01#0.01

lambda0 = 0.1

id = 0
ex = 'ex'
seed=0
display=1

def config():
    global id,ex,seed,display,Nx1,alpha_r1,alpha_b1,beta_b1,beta_r1,Nx2,Ny2,alpha_r2,alpha_b2,beta_b2,beta_r2,beta,alpha1,tau1,alpha2,tau2,sigma1,sigma2,kx,kr,tauc,U0,lambda0,alpha_nmf,l1_ratio_nmf,gamma
    args = sys.argv
    for s in args:
        id      = arg2i(id,"id=",s)
        ex      = arg2a(ex, 'ex=', s)
        seed    = arg2i(seed,"seed=",s)
        display = arg2i(display,"display=",s)

        Nx1      = arg2i(Nx1, 'Nx1=', s)
        alpha_r1 = arg2f(alpha_r1,"alpha_r1=",s)
        alpha_b1 = arg2f(alpha_b1,"alpha_b1=",s)
        beta_r1  = arg2f(beta_r1,"beta_r1=",s)
        beta_b1  = arg2f(beta_b1,"beta_b1=",s)
        beta    = arg2f(beta,"beta=",s)
        alpha1  = arg2f(alpha1,"alpha1=",s)

        Nx2      = arg2i(Nx2, 'Nx2=', s)
        Ny2      = arg2i(Ny2, 'Ny2=', s)
        alpha_r2 = arg2f(alpha_r2,"alpha_r2=",s)
        alpha_b2 = arg2f(alpha_b2,"alpha_b2=",s)
        beta_r2  = arg2f(beta_r2,"beta_r2=",s)
        beta_b2  = arg2f(beta_b2,"beta_b2=",s)
        #beta    = arg2f(beta,"beta=",s)
        alpha2  = arg2f(alpha2,"alpha2=",s)
        
        tau1     = arg2f(tau1,"tau1=",s)
        sigma1  = arg2f(sigma1,"sigma1=",s)
        tau2     = arg2f(tau2,"tau2=",s)
        sigma2  = arg2f(sigma2,"sigma2=",s)
        kx      = arg2i(kx,"kx=",s)
        kr      = arg2i(kr,"kr=",s)

        tauc    = arg2f(tauc,"tauc=",s)
        U0      = arg2f(U0,"U0=",s)

        lambda0 = arg2f(lambda0, 'lambda0=', s)
        gamma = arg2f(gamma, 'gamma=', s)
        alpha_nmf = arg2f(alpha_nmf, 'alpha_nmf=',s)
        l1_ratio_nmf = arg2f(l1_ratio_nmf, 'l1_ratio_nmf',s)
        
def output():

    str="%d,%d,%d,%f,%f,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (id,seed,Nx1,alpha_r1,alpha_b1,beta_b1,beta_r1,Nx2,Ny2,alpha_r2,alpha_b2,beta_b2,beta_r2,beta,alpha1,tau1,alpha2,tau2,sigma1,sigma2,kx,kr,tauc,U0,lambda0, alpha_nmf,l1_ratio_nmf,gamma, RMSE1,RMSE2,RMSE3,NRMSE1,NRMSE2,NRMSE3)
    
    #print(str)
    filename= 'data_hpcrc07_' + ex + '.csv'
    f=open(filename,"a")
    f.write(str)
    f.close()


def generate_data_sequence(T2):
    global Ny1
    global D1
    global A
    t0=1838
    #f=open('cello.txt','r')
    f=open('phrase1.txt','r')
    n=0
    D1 = np.zeros((T2*t0, Ny1))
    A = np.zeros(t0*Ny1)
    for line in f:
        A[n]=line
        n=n+1
    f.close()
    for h in range(T2):
        for i in range(Ny1):
            for j in range(t0):
                # print(A[j*Ny1+i])
                D1[j+h*t0,i]=A[j*Ny1+i]/500000
            #print(D1)
            
        
def generate_weight_matrix():
    global Wr1, Wb1, Wo1, Wr2, Wb2, Wo2
    ### Wr1
    Wr0 = np.zeros(Nx1 * Nx1)
    nonzeros = Nx1 * Nx1 * beta_r1
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx1, Nx1))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr1 = Wr0 / lambda_max * alpha_r1
    E = np.identity(Nx1)
    Wr1 = Wr1 + alpha1*E

    #print("lamda_max", lambda_max)
    #print("Wr1:")
    #print(Wr1)

    ### Wb1
    Wb1 = np.zeros(Nx1 * Ny1)
    Wb1[0:int(Nx1 * Ny1 * beta_r1 / 2)] = 1
    Wb1[int(Nx1 * Ny1 * beta_r1 / 2):int(Nx1 * Ny1 * beta_r1)] = -1
    np.random.shuffle(Wb1)
    Wb1 = Wb1.reshape((Nx1, Ny1))
    Wb1 = Wb1 * alpha_b1
    #print("Wb1:")
    #print(Wb1)

    ### Wo1
    Wo1 = np.ones(Ny1 * Nx1)
    Wo1 = Wo1.reshape((Ny1, Nx1))
    Wo1 = Wo1
    #print(Wo1)

    ### Wr2
    Wr0_2 = np.zeros(Nx2 * Nx2)
    nonzeros = Nx2 * Nx2 * beta_r2
    Wr0_2[0:int(nonzeros / 2)] = 1
    Wr0_2[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0_2)
    Wr0_2 = Wr0_2.reshape((Nx2, Nx2))
    v = scipy.linalg.eigvals(Wr0_2)
    lambda_max = max(abs(v))
    Wr2 = Wr0_2 / lambda_max * alpha_r2
    E = np.identity(Nx2)
    Wr2 = Wr2 + alpha2*E

    #print("lamda_max", lambda_max)
    #print("Wr2:")
    #print(Wr2)

    ### Wb2
    Wb2 = np.zeros(Nx2 * Ny2)
    Wb2[0:int(Nx2 * Ny2 * beta_r2 / 2)] = 2
    Wb2[int(Nx2 * Ny2 * beta_r2 / 2):int(Nx2 * Ny2 * beta_r2)] = -1
    np.random.shuffle(Wb2)
    Wb2 = Wb2.reshape((Nx2, Ny2))
    Wb2 = Wb2 * alpha_b2
    #print("Wb2:")
    #print(Wb2)

    ### Wo2
    Wo2 = np.ones(Ny2 * Nx2)
    Wo2 = Wo2.reshape((Ny2, Nx2))
    Wo2 = Wo2
    #print(Wo2)

    
def fx(x):
    #return np.tanh(x)
    return (np.tanh(x*beta)+1.0)/2.0
    #return np.fmax(x,0)-np.fmax(x-1,0)

theta=1
eps0=0.01
mu=0.0
#y=np.log(1+np.exp(theta*x))/theta
def fy(x):
    #return np.tanh(x)
    #return np.log(np.exp(theta*x)+1.0+eps0)/theta
    #return 1.0/(1.0+np.exp(-x))
    #return (np.tanh((x-mu)/theta)+1.0)*0.5
    return np.fmax(x-mu,0)
def fyi(x):
    #return np.arctanh(x)
    #return np.log(np.exp(theta*x)-1.0+eps0)/theta
    #return -np.log(1.0/(x+0.0001)-1.0)
    #return mu + theta * np.arctanh(2.0*x-1.0+0.001)
    return x+mu

def fr(x):
    return np.fmax(0, x)

def nmf():
    global U,invU,D2,Y1
    #np.set_printoptions(threshold=np.inf)
    print(Y1.shape)
    k=Ny2
    model = NMF(n_components=k, init='random', random_state=0,alpha=alpha_nmf,l1_ratio=l1_ratio_nmf)
    U = model.fit_transform(Y1.T)
    D2_T = model.components_
    print(np.sum(U))
    print(np.sum(D2_T))
    invU=np.linalg.pinv(U)
    R_re=U@D2_T
    D2_re=U.T@Y1.T
    D2=D2_T.T
    sum=0
    for i in range (Ny2):
        #            sum=np.linalg.norm(R1.T[i,:]-R_re[i,:])
        sum=np.linalg.norm(D2_T[i,:]-D2_re[i,:])
    print(sum/Ny2)


def run_network(Ttotal,Ttf,Ted,mode):

    
    global X1, Y1, R1 ,X2 ,Y2, R2, D2,S1,U
    X1 = np.zeros((Ttotal, Nx1))
    Y1 = np.zeros((Ttotal, Ny1))
    R1 = np.zeros((Ttotal, Ny1))
    S1 = np.zeros((Ttotal, Ny1))
    X2 = np.zeros((Ttotal, Nx2))
    Y2 = np.zeros((Ttotal, Ny2))
    R2 = np.zeros((Ttotal, Ny2))
    S2 = np.zeros((Ttotal, Ny2))
    D2 = np.zeros((Ttotal, Ny2))
    n = 0
    x1 = np.random.uniform(-1, 1, Nx1) * 0.0 + 0.0
    c1 = np.ones(Nx1)
    y1 = np.zeros(Ny1)
    #d = np.zeros(Ny1)
    r1 = np.zeros(Ny1)
    s1 = np.zeros(Ny1)
    x2 = np.random.uniform(-1, 1, Nx2) * 0.0 + 0.0
    c2 = np.ones(Nx2)
    y2 = np.zeros(Ny2)
    #d = np.zeros(Ny1)
    r2 = np.zeros(Ny2)
    s2 = np.zeros(Ny2)
    X1[n, :] = x1
    Y1[n, :] = y1
    R1[n, :] = r1
    S1[n, :] = s1
    X2[n, :] = x2
    Y2[n, :] = y2
    R2[n, :] = r2
    S2[n, :] = s2
    for n in range(Ttotal - 1):

        ### Lower area ###
        
        if n >= kx:
            xd1 = X1[n - kx, :]
        else:
            xd1 = np.zeros(Nx1)

        if n >= kr:
            if n < Ttf:
                yd1 = D1[n - kr, :]
            else:
                if mode==1 :
                    """
                    yd1 = arctanh(Y2[n - kr, :]@U.T)
                    for j in range(Ny2):
                        if Y2[n - kr, j]@U.T>1.0:
                            yd1[j]=1
                        if Y2[n - kr, j]@U.T<-1.0:
                            yd1[j]=-1
                    """
                    yd1 = gamma*U@Y2[n - kr, :]
                else:
                    yd1 = Y1[n - kr, :]
        else:
            yd1 = np.zeros(Ny1)

        if n >= kr:
            if n < Ted:
                yr1 = S1[n - kr, :]
            else:
                yr1 = np.zeros(Ny1)
        else:
            yr1 = np.zeros(Ny1)

        sum = np.zeros(Nx1)

        #sum += Wr1@xd
        sum += Wr1@(2.0*xd1-1.0)
        #sum += Wr1@(2.0*xd*c-1.0)
        #sum += Wr1@(xd*c)
        sum += Wb1@yd1
        sum += Wb1@yr1

        _x1 = x1 + 1.0 / tau1 * (-x1 + fx(sum) )
        #x += 1.0 / tau1 * np.random.normal(0,sigma,Nx1)
        y1 = fy(Wo1@_x1)
        #y = fy(Wo1@(2.0*_x-1.0))
        d1 = D1[n, :]
        r1 = d1 - y1
        s1 = fr(r1)

        # update and record
        x1=_x1
        X1[n + 1, :] = _x1
        Y1[n + 1, :] = y1
        R1[n + 1, :] = r1
        S1[n + 1, :] = s1
        #print(y)
        
        if mode==1:
            ### Higher area ###
        
            #            D2[n , :] = tanh(y1@U)
            D2[n , :] = (1.0/gamma)*U.T@y1
            if n >= kx:
                xd2 = X2[n - kx, :]
            else:
                xd2 = np.zeros(Nx2)

            if n >= kr:
                if n < Ttf:
                    yd2 = D2[n - kr, :]
                else:
                    yd2 = Y2[n - kr, :]
            else:
                yd2 = np.zeros(Ny2)

            if n >= kr:
                if n < Ted:
                    yr2 = S2[n - kr, :]
                else:
                    yr2 = np.zeros(Ny2)
            else:
                yr2 = np.zeros(Ny2)

            sum2 = np.zeros(Nx2)

            #sum2 += Wr2@xd2
            #sum2 += Wr2@(2.0*xd2-1.0)
            sum2 += Wr2@(2.0*xd2-1.0)
            #sum2 += Wr2@(xd2*c2)
            sum2 += Wb2@yd2
            sum2 += Wb2@yr2

            _x2 = x2 + 1.0 / tau2 * (-x2 + fx(sum2) )
            #x2 += 1.0 / tau2 * np.random.normal(0,sigma,Nx2)
            y2 = fy(Wo2@_x2)
            #y2 = fy(Wo2@(2.0*_x2-1.0))
            d2 = D2[n, :]
            r2 = d2 - y2
            s2 = fr(r2)

            # update and record
            x2=_x2
            X2[n + 1, :] = _x2
            Y2[n + 1, :] = y2
            R2[n + 1, :] = r2
            S2[n + 1, :] = s2
            #print(y)

        
        
def update_network(n):
    global Wo1, Wo2, D2

    if n==0 :
        run_network(T1,T1,T1,0) #Separated run mode
        

        #Nx12=50
        M = X1[T0:,:]
        invD1 = fyi(D1)
        G = invD1[T0:, :]

        ### Ridge regression
        E = np.identity(Nx1)
        TMP1 = inv(M.T@M + lambda0 * E)
        WoT = TMP1@M.T@G
        Wo1 = WoT.T
        run_network(T1,T1,T1,0) #Separated run mode
        nmf()
        
    else:
        run_network(T1,T1,T1,1) #Jointed run mode

        M = X1[T0:,:]
        invD1 = fyi(D1)
        G = invD1[T0:, :]

        ### Ridge regression
        E = np.identity(Nx1)
        TMP1 = inv(M.T@M + lambda0 * E)
        WoT = TMP1@M.T@G
        Wo1 = WoT.T
        
        #Nx12=50
        M2 = X2[T0:,:]
        invD2 = fyi(D2)
        G2 = invD2[T0:, :]

        ### Ridge regression
        E = np.identity(Nx2)
        TMP2 = inv(M2.T@M2 + lambda0 * E)
        WoT2 = TMP2@M2.T@G2
        Wo2 = WoT2.T

    
def train_train():
    global sigma
    sigma = sigma1
    for i in range (10):
        update_network(i)

def test_network():
    global sigma
    sigma = sigma2
    run_network(T2,T3,T3*3,1)

def analyze():
    global rmse
    rmse=np.sqrt(np.sum(R1*R1)/R1.size)
    print("rmse:",rmse)

def plot():
    plt.subplot(3,2,1)
    plt.plot(X1)
    plt.subplot(3,2,3)
    plt.ylim(0,1)
    plt.plot(Y1)
    plt.subplot(3,2,5)
    plt.plot(D1)


    plt.subplot(3,2,2)
    plt.plot(X2)
    plt.subplot(3,2,4)
    plt.plot(Y2)
    plt.subplot(3,2,6)
    plt.plot(D2)
    
    plt.show()


def plot_colormap():

    x=arange(T1)
    y=arange(Ny1)
    y2=arange(Ny2)
    X,Y=meshgrid(x,y)
    X2,Y2=meshgrid(x,y2)
    #pcolor(X, Y, np.log(Y1.T+0.1))
    pcolor(X, Y, np.log(Y1.T+0.0000001))
    colorbar()
    show()

    #X,Y=meshgrid(x,y)
    #pcolor(X, Y, np.log(Y1.T+0.1))
    pcolor(X, Y, np.log(D1.T+0.0000001))
    colorbar()
    show()

    pcolor(X2, Y2, np.log(D2.T+0.0000001))
    colorbar()
    show()

    
def execute():
    global T0,T1,T2,T3
    global D1
    global RMSE
    global RMSE1,RMSE2,RMSE3
    global NRMSE
    global NRMSE1,NRMSE2,NRMSE3

    Ny1 = 256

    np.random.seed(seed)

    generate_weight_matrix()

    #t0 = 3789
    t0 = 1838
    #t0 = 10940
    T0 = t0 # length of transient,
    T1 = t0*4 # length of training data
    T3 = t0
    Ntest = 1 # 50
    T2 = T0 + 3*T3 # length of test data

    generate_data_sequence(4)

    train_train()

    SUM = np.zeros(3)
    for i in range(Ntest):
        #D=DD[T1 + T2*i : T1 + T2*(i+1)]
        test_network()
        
        SUM_rmse=np.zeros(3)
        SUM_nrmse=np.zeros(3)
        rho1=np.std(Y1[:T1,:])
        rho2=np.std(Y1[T0+T3:T0+2*T3,:])
        rho3=np.std(Y1[T0+2*T3:T0+3*T3,:])
        for j in range(0,T3):
            SUM_rmse[0] += np.linalg.norm(Y1[j,:] - D1[j,:])
            SUM_nrmse[0] += np.linalg.norm(Y1[j,:] - D1[j,:])/rho1
        for j in range(T0+T3,T0+2*T3):
            SUM_rmse[1] += np.linalg.norm(Y1[j,:] - D1[j,:])
            SUM_nrmse[1] += np.linalg.norm(Y1[j,:] - D1[j,:])/rho2
        for j in range(T0+2*T3,T0+3*T3):
            SUM_rmse[2] += np.linalg.norm(Y1[j,:] - D1[j,:])
            SUM_nrmse[2] += np.linalg.norm(Y1[j,:] - D1[j,:])/rho3

    RMSE = SUM_rmse/np.sqrt(T3)
    NRMSE = SUM_nrmse/np.sqrt(T3)
    RMSE1=RMSE[0]
    RMSE2=RMSE[1]
    RMSE3=RMSE[2]
    NRMSE1=NRMSE[0]
    NRMSE2=NRMSE[1]
    NRMSE3=NRMSE[2]
    print(RMSE[0],RMSE[1],RMSE[2],NRMSE[0],NRMSE[1],NRMSE[2])


    if display :
        plot()
        plot_colormap()

if __name__ == "__main__":
    config()
    execute()
    output()
