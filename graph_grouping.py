#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#csv_data=pd.read_csv("model_v:1.0_e:testpandas_data_pcrc1.csv",sep=" ",header=None)
csv_data=pd.read_csv("data_ex2.csv",sep=",",names=['a','b','c','d','e','f','g','h','i'])
#csv_data2=csv_data.drop([0,1],axis=1) 
csv_data=csv_data.fillna(1)
print(csv_data)
means=csv_data['h'].groupby([csv_data['c']]).mean()
mins=csv_data['h'].groupby([csv_data['c']]).min()
#std=csv_data['e'].groupby([csv_data['b']]).std(ddof=False)
#x=np.arange(0,1.01,0.05)
#x=np.arange(0.5,10.6,0.1)
#RMSE=csv_data['g']
x=np.arange(0,1.01,0.05)
print(means)

plt.plot(x,means,label="means",linestyle='solid')
#plt.errorbar(x,means,yerr=std,label="standaed_deviation",fmt='',ecolor='r')
plt.legend()
plt.xlabel(u"Smoothnes")
plt.ylabel(u"RMSE1")
plt.title(u"Analyze smooting")
plt.axis([0,1,0,1])
plt.show()
plt.savefig( 'graph.png' )
