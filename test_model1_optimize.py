# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:

import optimization as opt
import searchtools as st

opt.parallel=8
opt.exe="python ed_hpcrc10.py seed=1 ex=opt11 display=0"
opt.filename_tmp="data_ed_hpcrc10_opt1.csv"
opt.filename_rec="data_model1_opt1_rec.csv"
opt.columns=['id','seed','Nx1','alpha_r1','alpha_b1','beta_b1','beta_r1','Nx2','Ny2','alpha_r2','alpha_b2','beta_b2','beta_r2','beta','alpha1','tau1','alpha2','tau2','sigma1','sigma2','kx','kr','tauc','U0','lambda0', 'alpha_nmf','l1_ratio_nmf','gamma', 'RMSE1','RMSE2','RMSE3','NRMSE1','NRMSE2','NRMSE3','NRMSE_all']
opt.clear()#設定をクリアする
opt.appendid()#id:必ず加える
opt.appendseed()# 乱数のシード（０から始まる整数値）
opt.append("alpha_r1",value=1.0,min=-5,max=5,round=3)#([変数名],[基本値],[下端],[上端],[まるめの桁数])
opt.append("alpha_r2",value=1.0,min=-5,max=5,round=3)
opt.minimize({'target':"NRMSE2",'iteration':10,'population':2,'samples':2})

"""
opt.clear()
opt.appendid()
opt.appendseed("seed")# 乱数のシード
opt.appendxfr("alpha_r1",0.3,0.001,2,2)
opt.appendxfr("alpha_r2",0.3,0.001,2,2)
opt.appendxfr("alpha_b1",0.1,0.001,1,2)
opt.appendxfr("alpha_b2",0.1,0.001,1,2)
opt.appendxfr("alpha_nmf",0.7,0,1,2)
opt.config()

#opt.names=['id','seed','x1','x2','y1','y2','y3']
#opt.names=['id','seed','alpha_r1','alpha_r2','alpha_b1','alpha_b2','NRMSE']
opt.names=['id','seed','Nx1','alpha_r1','alpha_b1','beta_b1','beta_r1','Nx2','Ny2','alpha_r2','alpha_b2','beta_b2','beta_r2','beta','alpha1','tau1','alpha2','tau2','sigma1','sigma2','kx','kr','tauc','U0','lambda0', 'alpha_nmf','l1_ratio_nmf','gamma', 'RMSE1','RMSE2','RMSE3','NRMSE1','NRMSE2','NRMSE3','NRMSE_all']
opt.minimize({'target':"NRMSE2",'iteration':10,'population':20,'samples':1})
"""
