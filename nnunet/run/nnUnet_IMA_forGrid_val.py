# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:20:18 2019

@author: liang
"""
import sys
sys.path.append('../../core')
#%%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import pandas as pd
import time
#%%
random_seed=1
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)

#%%

#%% ------ use this line, and then this file can be used as a Python module --------------------
def grid_validation(noise, delta, task):
#%%

    loss_name = "model_IMA"+task+"_N_"+str(noise)+"_D_"+str(delta)+"_Dice_epoch50_result_wba_L2_"+task+".pt"

    acc_list = []
    noise_list = []
    auc = 0
    for n in os.listdir("./result_grid"):
        if loss_name == n:
            print ("selected filename is ",n)
            checkpoint=torch.load("./result_grid/"+n, map_location=torch.device('cpu'))
            # extract the acc_lsit and noise level list to get the auc and plots
            #print ("good")
            
            acc_list = []

            noise_list = []
            for i in checkpoint['result_100pgd']:
                #if i['noise_norm']<0.3:
                #    continue
                acc_list.append(i['avgDice'])
                noise_list.append(i['noise'])
            auc = cal_AUC_robustness(acc_list, noise_list, noise_list[-1])
            print ("auc robustness is ", auc)
    return auc, acc_list, noise_list
            
#%%
def cal_AUC_robustness(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    
    # times acc clean and sqrt 
    auc_robust = np.sqrt(acc_list[0]*auc)
    return auc_robust

def cal_AUC(acc_list, noise_level_list, maxNoise):
    #get the acc at the largest noise level
    return acc_list[-1]


def cal_AUC_old(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    

    return auc

def cal_AUC_robustness_2_old(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    
    # times acc clean and sqrt 
    auc_robust = np.sqrt(acc_list[0]*acc_list[0]*auc)
    return auc_robust

def cal_AUC_robustness_2(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data

    return np.sqrt(acc_list[0]*acc_list[-1])



def get_heat_map(data, epsilons, deltas, path):
    #heatmap = np.array(data)
    #plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(8,7))
    plt.rc('text', usetex=True)
    df = pd.DataFrame(data, index = epsilons, columns = deltas)
    df["mean"] = df.mean(axis = 1)
    df.loc["mean"] = df.mean(axis = 0)
    df_v = df.copy()
    df_v["mean"] = float("nan")
    df_v.loc["mean"] = float("nan")
    df_m = df.copy()
    df_m.loc[:-1,:-1] = float("nan")
    df_m.loc["mean","mean"] = float("nan")
    sns.heatmap(ax = ax, data = df_v, annot=True, fmt = ".4f", cmap="Reds",)
    sns.heatmap(ax = ax, data = df_m, annot=True, fmt = ".4f", cmap="Blues",)
    ax.set(ylabel=''r'$\epsilon$', xlabel=''r'$\Delta_\epsilon$')
    ax.figure.savefig(path+".pdf", bbox_inches='tight',pad_inches=0.0)
    ax.figure.clf()

if __name__=="__main__":
    #[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    #[0.005,0.01, 0.015, 0.02, 0.025, 0.03 ]
    # in paper it is 1.5
    import matplotlib.pyplot as plt
    import seaborn as sns
    choice = 2
    task = ""
    mat = []
    matacc=[]
    matrobust = []
    mat2 = []
    
    if choice == 0:  
        epsilons = [15,20,25,30,35]
        deltas = [4,5,6]
        task = "002"
        
    if choice == 1:
        task = "004"
        epsilons = [9,12,15,18]
        deltas = [1.5,2.0,2.5]

    if choice == 2:
        task = "005"
        epsilons = [20,30,40,50]
        deltas = [5,10,15]        
    
    
    
    
    for i, noise in enumerate(epsilons):
        line = []
        lineacc = []
        linerobust = []
        line2 = []
        for j, delta in enumerate(deltas): # in paper it is 1.5/100
            auc, accList, noiseList = grid_validation(noise, delta, task)
            line.append(round(auc, 4))
            lineacc.append(round(accList[0],4))
            linerobust.append(round(cal_AUC(accList, noiseList, noiseList[-1]), 4))
            line2.append(round(cal_AUC_robustness_2(accList, noiseList, noiseList[-1]),4))
        mat.append(line)
        matacc.append(lineacc)
        matrobust.append(linerobust)
        mat2.append(line2)
    
    
    
    
    datas = [matacc, matrobust, mat, mat2]
    paths = ["clean_AVG_Dice","noisy_AVG_Dice", "AUC_Robust", "GeoMean"]
    
    
    
    for i in range(4):
        
        get_heat_map(datas[i], epsilons, deltas, task+paths[i])
        
