# Copyright (c) 2017-2018 Yuichi Katori All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: pandas, CSVファイルを読み込んで、散布図を作成する。
'''
 使う前に、
 $ python test_pandas_generate_testdata.py
 としてテストデータを生成してください。
'''


import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#names=['id','seed','sm','alpha_r1','alpha_r2','gamma_r1','gamma_r2','alpha_nmf','l1_ratio_nmf','NRMSE_1','NRMSE_2','tau1','tau2']
names=['id','seed','Nx1','alpha_r1','alpha_b1','beta_b1','beta_r1','Nx2','Ny2','alpha_r2','alpha_b2','beta_b2','beta_r2','beta','alpha1','tau1','alpha2','tau2','sigma1','sigma2','kx','kr','tauc','U0','lambda0','alpha_nmf','l1','gamma','RMSE_0','RMSE_1','RMSE_2','NRMSE_0','NRMSE_1','NRMSE_2','NRMSE_all']
df=pd.read_csv("data_ex100_hpcrc10.csv",sep=',',names=names)
#print(df)

# 単一の散布図
#df.plot(x='Nx1', y='Nx2', kind='scatter')

# 散布図の行列
#df2=df.drop('id',axis=1) # 指定した列を削除

df2=df[['alpha_r1','alpha_r2','NRMSE_1']] # 指定した列のみでデータフレームを構成する
print(df2)
"""
df2b=df2['RMSE_2']*np.sqrt(df2['NRMSE_2']/df2['RMSE_2']) # TODO
df2=df2.drop("NRMSE_2",axis=1)
df2=pd.concat([df2,df2b],axis=1)
df2 = df2.rename(columns={0: 'NRMSE_2'})
"""
scatter_matrix(df2,alpha=0.8, figsize=(3, 3), diagonal='kde')

# 条件を満たすデータについての散布図
#df3 = df2[ (df2['NRMSE_2']<=5000) ]

#scatter_matrix(df3, c='red',alpha=0.8, figsize=(4, 4), diagonal='kde')
"""
df4 = df2[ (df2['NRMSE_1']<=180) ]
#df5 = df2[ (df2['RMSE_2']<4) & (df2['NRMSE_2']<=500) ]
#print(df5)
scatter_matrix(df4, c='green', alpha=0.8, figsize=(4, 4), diagonal='kde')
#scatter_matrix(df5, alpha=0.8, figsize=(6, 6), diagonal='kde')
"""

plt.show()
