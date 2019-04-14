#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Katori Lab All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# DONE: fft の時系列のうち振幅情報のみを用いて（位相情報を使わず）、元の音声を再-構成する。

import numpy as np
import sys
np.set_printoptions(threshold=np.inf)
import scipy.io.wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal

wavfile = "Cello1.wav"
fs, x = scipy.io.wavfile.read(wavfile)

y=fft(x)

L=len(x) # length of given data
N=512 # number of points in window function
S=256 # shift
M=int((L-N)/S)

w=signal.hann(N) # window function

Yang=np.zeros(N*M).reshape((N,M))
Yabs=np.zeros(N*M).reshape((N,M))
Yreal=np.zeros(N*M).reshape((N,M))
Yimag=np.zeros(N*M).reshape((N,M))
Y=np.ndarray(N*M,dtype=complex).reshape((N,M))

#print(Yabs.shape)
for i in range(0,M):
    s=0+i*S
    xc=x[s:s+N]
    xw=xc*w
    y=fft(xw)

    #Y[:,i]=y
    #Y[:,i]=y.real
    Y[:,i]=np.absolute(y)

    Yreal[:,i]=y.real
    Yimag[:,i]=y.imag
    Yabs[:,i]=np.absolute(y)
    Yang[:,i]=np.angle(y)

for i in range(0,M):
    for j in range(0,256):
       print(Yabs[j,i])#,end=" ")
