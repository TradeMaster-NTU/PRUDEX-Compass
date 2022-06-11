import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import seaborn as sns
sns.set_style("white")
import matplotlib.patches as mpatches
import collections
import os
"""the prepartion process is very similiar to the distribution.py where the distribution need 
to generate one dictionary but here we need to generate multiple dictionary and combine them all together.
here is an example
"""

def tt(ret_lst):
    """Total Return"""
    net_value = 1
    for ret in ret_lst:
        net_value *= (1+ret)
    return net_value-1

def vol(ret_lst):
    """Volatility"""
    return np.std(ret_lst)

def dd(ret_lst):
    """Downdown Deviation"""
    neg_ret_lst = []
    for ret in ret_lst:
        if ret < 0:
            neg_ret_lst.append(ret)
    return np.std(neg_ret_lst)

def mdd(ret_lst):
    """Maximum Drawdown"""
    net_value_lst = [1]
    net_value = 1
    for ret in ret_lst:
        net_value *= (1+ret)
        net_value_lst.append(net_value)
    mdd = 0
    peak = net_value_lst[0]
    for net_value in net_value_lst:
        if net_value > peak:
            peak = net_value
        dd = (peak - net_value)/peak
        if dd > mdd:
            mdd = dd
    return mdd

def cr(ret_lst):
    """Calmar Ratio"""
    return tt(ret_lst)/mdd(ret_lst)

def sor(ret_lst):
    """Sortino Ratio"""
    return tt(ret_lst)/dd(ret_lst)/len(ret_lst)

def sr(ret_lst):
    """Sharpe Ratio"""
    mean = np.mean(ret_lst)
    std = np.std(ret_lst)
    return mean/std * np.sqrt(len(ret_lst))  
algs = ['A2C','PPO','SAC','SARL','DeepTrader','AlphaMix+']
Entropy_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/Entropy.npy")
        data2=np.load("./"+a+"/"+market+"/Entropy1.npy")
        data3=np.load("./"+a+"/"+market+"/Entropy2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    Entropy_dict[a]=np.array(contents)


tt_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/tt.npy")
        data2=np.load("./"+a+"/"+market+"/tt1.npy")
        data3=np.load("./"+a+"/"+market+"/tt2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    tt_dict[a]=np.array(contents)

sr_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/sr.npy")
        data2=np.load("./"+a+"/"+market+"/sr1.npy")
        data3=np.load("./"+a+"/"+market+"/sr2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    sr_dict[a]=np.array(contents)


cr_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/cr.npy")
        data2=np.load("./"+a+"/"+market+"/cr1.npy")
        data3=np.load("./"+a+"/"+market+"/cr2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    cr_dict[a]=np.array(contents)

sor_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/sor.npy")
        data2=np.load("./"+a+"/"+market+"/sor1.npy")
        data3=np.load("./"+a+"/"+market+"/sor2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    sor_dict[a]=np.array(contents)



vol_dict={}
for a in ["A2C","SARL","DeepTrader","PPO","AlphaMix+","SAC"]:
    contents=[]
    for market in ["Crypto","dj30","FX","sz50"]:
        data1=np.load("./"+a+"/"+market+"/vol.npy")
        data2=np.load("./"+a+"/"+market+"/vol1.npy")
        data3=np.load("./"+a+"/"+market+"/vol2.npy")
        for data in [data1,data2,data3]:
            contents.append(np.array(data))
    contents=np.array(contents)
    vol_dict[a]=np.array(contents)



dmc_scores = {}
dmc_scores["TR"]=tt_dict
dmc_scores["SR"]=sr_dict
dmc_scores["CR"]=cr_dict
dmc_scores["SoR"]=sor_dict
dmc_scores["VOL"]=vol_dict
dmc_scores["Entropy"]=Entropy_dict

#notice that we do not show the data in this particular project so you need to replace the position of your own data the data is as following
"""the important thing is to construct the dmc_score, which is a dictionary consisting of 4 different dictionaries
it looks something like this

{'TR': {'A2C': array([[ 1.60122609e+00,  1.04696694e-01,  1.46063288e-01,
           3.50520029e+00,  9.41977202e+00],
         [ 1.63091274e+00,  2.29133049e-01,  8.02841313e-02,
           9.95249730e-01,  3.27393262e-01],
         [ 4.59269848e-01,  3.15380462e-01,  1.71711564e+00,
           5.85607877e-01,  1.11524528e+00],
         [ 1.84490701e-01,  1.09170296e-01,  1.14688414e-02,
           1.84552353e-01,  1.36585675e-01],
         [ 5.94387876e-02,  1.21122098e-01,  1.31075271e-01,
           8.06887659e-03,  3.04353591e-02],
         [ 3.70925527e-01,  1.71984808e-01,  2.33636696e-01,
           1.09770307e-01,  1.43709447e-01],
         [-3.33345918e-02,  2.11417378e-02, -4.03713809e-02,
           3.58939090e-02, -2.98719592e-02],
         [-3.18310747e-02, -4.23086289e-02, -6.03686289e-02,
          -9.65202722e-02, -4.15505174e-02],
         [ 1.00168364e-01,  5.21695252e-02,  5.82298035e-02,
           7.43798428e-02,  6.03867720e-02],
         [ 1.06773553e-01,  9.89134872e-02,  4.87085505e-02,
          -2.70655563e-02,  3.77080088e-02],
         [ 5.18332332e-01,  2.25009686e-01,  3.34616193e-01,
           2.39745519e-01,  2.78250359e-01],
         [-1.14333212e-01, -1.05130800e-01, -2.31335790e-01,
           7.06243760e-02,  5.23086887e-02]]),
...
 'SR': {'A2C': array([[ 1.44212684,  0.49366565,  0.60828794,  2.16109458,  2.24352926],
        
...
       
then we can simpliy use the following code to generate the graph
"""
# here every key in the dictionary represents a indicator's name and the value is a dictionary with the name of the algorithms as key and a 2-dimension np.array as the value
#(1d for different randomseed and 1d for multiple task here is the time rolling)

def subsample_scores_mat(score_mat, num_samples=3, replace=False):
  subsampled_dict = []
  total_samples, num_games = score_mat.shape
  subsampled_scores = np.empty((num_samples, num_games))
  for i in range(num_games):
    indices = np.random.choice(total_samples, size=num_samples, replace=replace)
    subsampled_scores[:, i] = score_mat[indices, i]
  return subsampled_scores

def get_rank_matrix(score_dict, n=100000, algorithms=None):
  arr = []
  if algorithms is None:
    algorithms = sorted(score_dict.keys())
  print(f'Using algorithms: {algorithms}')
  for alg in algorithms:
    arr.append(subsample_scores_mat(
        score_dict[alg], num_samples=n, replace=True))
  X = np.stack(arr, axis=0)
  num_algs, _, num_tasks = X.shape
  all_mat = []
  for task in range(num_tasks):
    task_x = -X[:, :, task]
    rand_x = np.random.random(size=task_x.shape)
    indices = np.lexsort((rand_x, task_x), axis=0)
    mat = np.zeros((num_algs, num_algs))
    for rank in range(num_algs):
      cnts = collections.Counter(indices[rank])
      mat[:, rank] = np.array([cnts[i]/n for i in range(num_algs)])
    all_mat.append(mat)
  all_mat = np.stack(all_mat, axis=0)
  return all_mat
indicator_list=['TR','SR','VOL','Entropy']
algs=['A2C','PPO','SAC','SARL','DeepTrader','AlphaMix+']
colors=['moccasin','aquamarine','#dbc2ec','salmon','lightskyblue','pink','orange']
def make_rank_plot(algs,indicator_list,dict,colors):
  mean_ranks_all = {}
  all_ranks_individual = {}
  for key in indicator_list:
    dmc_score_dict = dmc_scores[key]
    algs =  algs
    all_ranks = get_rank_matrix(dmc_score_dict, 200000, algorithms=algs)
    mean_ranks_all[key] = np.mean(all_ranks, axis=0)
    all_ranks_individual[key] = all_ranks
  color_idxs = [0, 1,2,3,4,5,6]
  DMC_COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))
  keys = algs
  labels = list(range(1, len(keys)+1))
  width = 1.0      
  fig, axes = plt.subplots(nrows=1,ncols=len(indicator_list), figsize=(8, 2 * 2))
  for main_idx, main_key in enumerate(indicator_list):
    ax = axes[main_idx]
    mean_ranks = mean_ranks_all[main_key]
    bottom = np.zeros_like(mean_ranks[0])
    for i, key in enumerate(algs):
      label = key if main_idx == 0 else None
      ax.bar(labels, mean_ranks[i], width, label=label, 
            color=DMC_COLOR_DICT[key], bottom=bottom, alpha=0.9)
      bottom += mean_ranks[i]
    yticks = np.array(range(0, 101, 20))
    ax.set_yticklabels(yticks, size='large')
    if main_idx in list(range(len(indicator_list)+1)):
      ax.set_xticks(labels)
    else:
      ax.set_xticks([])
    ax.set_title(main_key, size='x-large', y=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    left= True
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                  left=left, right=False, labeltop=False,
                  labelbottom=True, labelleft=left, labelright=False)

  fig.legend(loc='center right', fancybox=True, ncol=1, fontsize='large', bbox_to_anchor=(1.15, 0.35))
  fig.subplots_adjust(top=0.72, wspace=0.5, bottom=0)
  fig.text(x=-0.01, y=0.2, s='Fraction (in %)', rotation=90, size='xx-large')
  plt.savefig(dict,bbox_inches = 'tight')
make_rank_plot(algs,indicator_list,dict,colors)