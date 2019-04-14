# Copyright (c) 2016-2017 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: PCRCの基本的なコードを実装する。クラスは使わずに。

import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
from arg2x import *

#T1 = 1838
#T0 = 5

Nx = 100
Ny = 256

#sigma_np = -5
alpha_r = 0.2
alpha_b = 0.15
beta_r = 0.1
beta_b = 0.1
beta = 2.5
alpha1 = 0.1
tau = 6


sigma=0
sigma1=0.0000 # noise in training
sigma2=0 # noise in test

kx = 0
kr = 0

tauc=10
U0=0.01

lambda0 = 0.1

id = 0
ex = 'ex'
seed=0
display=1

def config():
    global id,ex,seed,display,Nx,alpha_r,alpha_b,beta_b,beta_r,beta,alpha1,tau,sigma1,kx,kr,tauc,U0,lambda0
    args = sys.argv
    for s in args:
        id      = arg2i(id,"id=",s)
        ex      = arg2a(ex, 'ex=', s)
        seed    = arg2i(seed,"seed=",s)
        display = arg2i(display,"display=",s)

        Nx      = arg2i(Nx, 'Nx=', s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        beta    = arg2f(beta,"beta=",s)
        alpha1  = arg2f(alpha1,"alpha1=",s)

        tau     = arg2f(tau,"tau=",s)
        sigma1  = arg2f(sigma1,"sigma1=",s)
        kx      = arg2i(kx,"kx=",s)
        kr      = arg2i(kr,"kr=",s)

        tauc    = arg2f(tauc,"tauc=",s)
        U0      = arg2f(U0,"U0=",s)

        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f\n" \
    % (id,seed,Nx,alpha_r,alpha_b,beta_r,beta_b,beta,alpha1,tau,sigma1,kx,kr,tauc,U0,lambda0,RMSE[0],RMSE[1],RMSE[2])
    #print(str)
    filename= 'data_pcrc5_' + ex + '.csv'
    f=open(filename,"a")
    f.write(str)
    f.close()

def generate_mackey_glass(tau,T1):
    '''
    tau: delaya, typical value of tau = 17,
    T1: length
    '''
    delta = 0.1 #time constant
    T0 = int(tau / delta)
    T2 = T0+1000
    y = np.zeros(T2+T1)

    for n in range(T2+T1-1):
        if n < T0-1:
            y[n + 1] = 1.0 #np.random.uniform(0.0, 1.0)
        else:
            y[n + 1] = y[n] + delta * (0.2 * y[n - T0] / (1 + pow(y[n - T0],10)) - 0.1 * y[n])

    #print(y)
    return y[T2:]

def generate_data_sequence_():
    global T1, Ny
    global D
    D = np.zeros((T1, Ny))
    c = np.linspace(0, 1, Ny)
    for n in range(T1):
        d = np.zeros(Ny)
        t = 0.1 * n -10
        d += np.sin(t + c) * np.exp(-0.1 * t**2) * 0.5
        t = 0.1 * n -30
        d += np.sin(t + c) * np.exp(-0.1 * t**2) * 0.5
        D[n, :] = d

def generate_data_sequence2():
    global T1, Ny
    global D
    D = np.zeros((T1, Ny))
    c = np.linspace(0, 1, Ny)
    for n in range(T1):
        d = np.zeros(Ny)
        t = 0.1 * n -10
        d += np.sin(t + c) * np.exp(-0.1 * t**2) * 0.5
        t = 0.1 * n -30
        d += np.sin(t + c) * np.exp(-0.1 * t**2) * 0.5
        D[n, :] = d

def generate_data_sequence(T2):
    global Ny
    global D
    global A
    f=open('phrase1.txt','r')
#    f=open('0615chord_half.txt','r')
    n=0
    D = np.zeros((T2*1838, Ny))
    A = np.zeros(1838*Ny)
    for line in f:
        A[n]=line
        n=n+1
    f.close()
    for h in range(T2):
        for i in range(Ny):
            for j in range(1838):
                # print(A[j*Ny+i])
                D[j+h*1838,i]=A[j*Ny+i]/700000
            #print(D)
            
        
def generate_weight_matrix():
    global Wr, Wb, Wo
    ### Wr
    Wr0 = np.zeros(Nx * Nx)
    nonzeros = Nx * Nx * beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx, Nx))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr = Wr0 / lambda_max * alpha_r
    E = np.identity(Nx)
    Wr = Wr + alpha1*E

    #print("lamda_max", lambda_max)
    #print("Wr:")
    #print(Wr)

    ### Wb
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_r / 2)] = 1
    Wb[int(Nx * Ny * beta_r / 2):int(Nx * Ny * beta_r)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb = Wb * alpha_b
    #print("Wb:")
    #print(Wb)

    ### Wo
    Wo = np.ones(Ny * Nx)
    Wo = Wo.reshape((Ny, Nx))
    Wo = Wo
    #print(Wo)

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


def run_network(Ttotal,Ttf,Ted):

    global X, Y, R
    X = np.zeros((Ttotal, Nx))
    Y = np.zeros((Ttotal, Ny))
    R = np.zeros((Ttotal, Ny))
    S = np.zeros((Ttotal, Ny))
    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.0 + 0.0
    c = np.ones(Nx)
    y = np.zeros(Ny)
    #d = np.zeros(Ny)
    r = np.zeros(Ny)
    s = np.zeros(Ny)
    X[n, :] = x
    Y[n, :] = y
    R[n, :] = r
    S[n, :] = s
    for n in range(Ttotal - 1):
        if n >= kx:
            xd = X[n - kx, :]
        else:
            xd = np.zeros(Nx)

        if n >= kr:
            if n < Ttf:
                yd = D[n - kr, :]
            else:
                yd = Y[n - kr, :]
        else:
            yd = np.zeros(Ny)

        if n >= kr:
            if n < Ted:
                yr = S[n - kr, :]
            else:
                yr = np.zeros(Ny)
        else:
            yr = np.zeros(Ny)

        sum = np.zeros(Nx)

        #sum += Wr@xd
        #sum += Wr@(2.0*xd-1.0)
        sum += Wr@(2.0*xd*c-1.0)
        #sum += Wr@(xd*c)
        sum += Wb@yd
        sum += Wb@yr

        _x = x + 1.0 / tau * (-x + fx(sum) )
        _c = c + (1.0 - c)/tauc - U0*c*x
        #x += 1.0 / tau * np.random.normal(0,sigma,Nx)
        y = fy(Wo@_x)
        #y = fy(Wo@(2.0*_x-1.0))
        d = D[n, :]
        r = d - y
        s = fr(r)

        # update and record
        x=_x
        c=_c
        X[n + 1, :] = _x
        Y[n + 1, :] = y
        R[n + 1, :] = r
        S[n + 1, :] = s
        #print(y)


def update_network():
    global Wo

    run_network(T1,T1,T1)

    #Nx2=50
    M = X[T0:,:]
    invD = fyi(D)
    G = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T

def train_train():
    global sigma
    sigma = sigma1
    for i in range (10):
        update_network()

def test_network():
    global sigma
    sigma = sigma2
    run_network(T2,T3,T3*3)

def analyze():
    global rmse
    rmse=np.sqrt(np.sum(R*R)/R.size)
    print("rmse:",rmse)

def plot():
    plt.subplot(3,1,1)
    plt.plot(X)
    plt.subplot(3,1,2)
    plt.plot(Y)
#    generate_data_sequence(4)
    plt.subplot(3,1,3)
    plt.plot(D)
    plt.show()

def execute():
    global T0,T1,T2,T3
    global D
    global RMSE

    Ny = 256

    np.random.seed(seed)

    generate_weight_matrix()


    T0 = 1838 # length of transient,
#    T1 = 10000 # length of training data
    T1 = 1838*4 # length of training data
    T3 = 1838
    Ntest = 50 # 50
    T2 = T0 + 3*T3 # length of test data

    generate_data_sequence(4)

    #y = generate_mackey_glass(17, T1 + T2*Ntest)
    #DD=np.zeros((T1+T2*Ntest,1))
    #DD[:,0]=np.tanh(y-1)
    #DD[:,0]=np.log(1+np.exp(y))
    #DD[:,0]=np.fmax(y-1,0)

    #D = DD[:T1]
    train_train()

    SUM = np.zeros(3)
    for i in range(Ntest):
        #D=DD[T1 + T2*i : T1 + T2*(i+1)]
        test_network()
        """
        for j in range(T0,T0+T3):
            SUM[0] += (Y[j] - D[j])**2
        for j in range(T0+T3,T0+2*T3):
            SUM[1] += (Y[j] - D[j])**2
        for j in range(T0+2*T3,T0+3*T3):
            SUM[2] += (Y[j] - D[j])**2

    RMSE = np.sqrt(SUM/Ntest/T3)
    print(RMSE[0],RMSE[1],RMSE[2])"""

    if display :
        plot()


if __name__ == "__main__":
    config()
    execute()
    output()
