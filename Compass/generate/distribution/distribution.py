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
"""
here we use the crypto data as an example, yet the overall data is too huge to upload.
The key point is to build a dictionary with the name of the algorithms as a key and 2-dimensional array as its contens
(1d for seed and 1d for task, in the following contents, the task is the multi-time rolling)
Alert: the location for the file is not uniformed so you have to replace it with your own files' position
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

crypto_return=pd.read_csv("./crypto2021.csv",index_col=0)[["TR"]]
crypto_return1=pd.read_csv("./crypto2020.csv",index_col=0)[["TR"]]
crypto_return2=pd.read_csv("./crypto2019.csv",index_col=0)[["TR"]]
crypto_average_return=(crypto_return.iloc[-1].values)*0.01
crypto_average_return1=(crypto_return1.iloc[-1].values)*0.01
crypto_average_return2=(crypto_return2.iloc[-1].values)*0.01

tt_dict_crypto={}
for x in ['a2c','ddpg','pg','ppo','sac']:
    tts=[]

    for experiment in ["experiment","experiment rolling1","experiment rolling2"]:
        new_tts=[]


        for i in [12345,23451,34512,45123,51234]:

            seed_result=np.array(pd.read_csv(os.path.join('./experiment with crypto',x,experiment,'results',x.upper()+' '+'result{}'.format(i)+'.csv'))["daily_return"].tolist())
            tt_value=tt(seed_result)
            vol_value=vol(seed_result)
            mdd_value=mdd(seed_result)
            cr_value=cr(seed_result)
            sor_value=sor(seed_result)
            sr_value=sr(seed_result)
            new_tts.append(tt_value)

        if experiment=="experiment":
            average_return=crypto_average_return
        elif experiment=="experiment rolling1":
            average_return=crypto_average_return1
        else:
            average_return=crypto_average_return2
        new_tts=list((new_tts/average_return-0.5)*100)
        new_tts=[0 if i<0 else i for i in new_tts]
        new_tts=[100 if i>100 else i for i in new_tts]
        tts.append(new_tts)
    tt_dict_crypto[x]=np.array(tts)
experiment=np.load("./crypto/Mixcrypto.npy")
experiment1=np.load("./crypto/Mixcrypto1.npy")
experiment2=np.load("./crypto/Mixcrypto2.npy")
tts=[]
for name in ["experiment","experiment1","experiment2"]:
    if name=='experiment':
        x=experiment
    elif name=='experiment1':
        x=experiment1
    else:
        x=experiment2
    new_tts=[]
    for y in x:
        tt_value=tt(y)
        vol_value=vol(y)
        mdd_value=mdd(y)
        cr_value=cr(y)
        sor_value=sor(y)
        sr_value=sr(y)
        new_tts.append(tt_value)
    if name=='experiment':
        average_return=crypto_average_return
    elif name=='experiment1':
        average_return=crypto_average_return1
    else:
        average_return=crypto_average_return2
    new_tts=list((new_tts/average_return-0.5)*100)
    new_tts=[0 if i<0 else i for i in new_tts]
    new_tts=[100 if i>100 else i for i in new_tts]
    tts.append(new_tts)
tt_dict_crypto["RLmix"]=np.array(tts)
tt_dict_crypto['A2C']=tt_dict_crypto.pop('a2c')
tt_dict_crypto['PPO']=tt_dict_crypto.pop('ppo')
tt_dict_crypto['SAC']=tt_dict_crypto.pop('sac')
tt_dict_crypto['SARL']=tt_dict_crypto.pop('ddpg')
tt_dict_crypto['DeepTrader']=tt_dict_crypto.pop('pg')
tt_dict_crypto['AlphaMix+']=tt_dict_crypto.pop('RLmix')
"""
the above part is the we compute the dictionary which looks like this
{'A2C': array([[ 15.23634509,   0.        ,   0.        ,  92.8071008 ,
         100.        ],
        [100.        ,  27.09725747,   0.        , 100.        ,
          60.15924039],
        [ 14.2335452 ,   0.        , 100.        ,  31.90319952,
         100.        ],
        [ 46.84551244,   7.30724224,   0.        ,  46.87787585,
          21.69851713],
        [ 13.91267483,  80.2388153 ,  90.94115108,   0.        ,
           0.        ],
        [100.        ,  19.65767834,  44.62806634,   0.        ,
           8.20552742],
        [  0.        , 100.        ,   0.        , 100.        ,
           0.        ],
        [  6.04062442,  24.48702278,  56.28279733, 100.        ,
          23.15231934],
        [ 81.2822601 ,  18.37421386,  26.31691149,  47.48341132,
          29.14386897],
        [ 61.45464841,  53.24998665,   0.84399842,   0.        ,
           0.        ],
        [100.        ,  24.13828219,  60.25245249,  28.99358132,
          41.68051375],
        [100.        , 100.        , 100.        ,   0.        ,
           0.        ]]),
 'PPO': array([[  0.        ,   0.        ,   0.        ,   0.        ,
         100.        ],
        [100.        ,  50.7364656 ,   4.90470416, 100.        ,
           0.        ],
        [  9.26607132,   0.        , 100.        ,  43.36398955,
         100.        ],
        [ 17.63377665,   0.        ,  53.53718168,   0.        ,
          32.39963146],
        [100.        ,  86.42913224, 100.        ,  19.43942472,
         100.        ],
        [100.        ,  26.78644983,  42.71559351,   0.        ,
           6.29173408],
        [  0.        ,   0.        ,   0.        , 100.        ,
           0.        ],
        [  0.68902878,   5.86134848,  55.39402489, 100.        ,
          31.76254786],
        [ 72.0830016 ,  17.39282549,  25.6190015 ,  35.96302666,
          45.74469716],
        [  0.        ,   0.        ,  39.53507212,  78.63250468,
         100.        ],
        [ 96.05032955,  29.43233997,  57.27610373,  14.69063079,
          45.09484497],
        [ 88.9726685 , 100.        , 100.        ,  26.93152593,
          21.05487896]]),
 'SAC': array([[  6.183908  , 100.        ,  90.97165583,   0.        ,
           0.        ],
        [  0.        ,  45.81237445,   0.        ,   0.        ,
         100.        ],
        [  0.        ,  36.11689347, 100.        ,   2.4124593 ,
         100.        ],
        [  0.        , 100.        ,  12.35046281,  26.48164842,
         100.        ],
        [100.        , 100.        ,  69.45351124,   0.        ,
           0.        ],
        [100.        ,  37.14712914,  25.65758672,  40.61025741,
          62.93727969],
        [  0.        ,   0.        ,   0.        , 100.        ,
          46.46120641],
        [ 32.6935097 ,  20.39840274,  31.76561881,   7.37794114,
         100.        ],
        [ 32.3107863 ,  39.04407081,  20.95415859, 100.        ,
          45.86946075],
        [100.        , 100.        ,  73.85598011,   0.        ,
           0.        ],
        [ 40.51073684,  57.32014909,   3.44104859,  76.8845489 ,
           0.        ],
        [  0.        ,  10.51550099, 100.        , 100.        ,
         100.        ]]),
 'SARL': array([[  0.        ,   0.        , 100.        , 100.        ,
           0.        ],
        [ 64.97719714,   0.        ,  71.68707726,  94.19332371,
          61.62335126],
        [ 20.7817162 ,  69.10540509,  19.06579716,  63.09722616,
           0.        ],
        [ 54.72277161,  44.40411033,  32.37474776,  48.85923901,
          24.92518116],
        [100.        ,  24.49583503,  70.50476537,   0.        ,
         100.        ],
        [ 62.8720803 ,  53.97056023,  53.58233021,  69.27014403,
          67.64083854],
        [ 73.59559882,  50.96221777, 100.        ,  41.48929261,
           0.        ],
        [ 39.92451033,   0.        ,  48.30038216,  57.57309712,
          38.95330592],
        [ 27.64473258,  46.42073268,  32.66731543,  54.85915836,
           0.        ],
        [ 81.36840287,  68.93226721, 100.        ,  97.33667879,
           0.        ],
        [ 84.17766194,  36.81555394,  43.91087056,  76.94499322,
          46.69575373],
        [100.        ,   0.        ,  61.26623244, 100.        ,
          59.97760323]]),
 'DeepTrader': array([[  0.        ,   0.        , 100.        , 100.        ,
          56.43924497],
        [  0.        ,  76.66269215, 100.        , 100.        ,
          89.70402915],
        [  0.        ,   0.        ,   0.        ,   0.        ,
         100.        ],
        [ 51.91345552,  63.17986088,  19.9673035 ,   0.        ,
          89.99735458],
        [100.        ,  26.13435858,   0.        , 100.        ,
           0.        ],
        [ 38.10891247,  65.93084229,  21.44068736,  43.669265  ,
          42.24121666],
        [  0.        ,   0.        ,   0.        ,   0.        ,
         100.        ],
        [100.        ,   0.        ,   0.        ,  61.26383646,
          47.61965027],
        [  0.        ,   5.97135089,  62.49130402,   0.        ,
          72.35731295],
        [ 40.70486831,   0.        ,  69.96972571, 100.        ,
         100.        ],
        [  0.        , 100.        ,  12.53544418,   4.8025416 ,
           0.        ],
        [  0.        ,   0.        ,   0.        ,   0.        ,
           0.        ]]),
 'AlphaMix+': array([[100.        ,  75.89386133,  65.08022268,  34.81956972,
          64.59159222],
        [ 66.24012374,  62.53597946,  51.72490752,  70.62768842,
          45.38165391],
        [ 25.17687293,  39.51481158,  24.92237526,  92.60292004,
          49.26650834],
        [ 46.56441078,  70.08116912,  81.14691157,  46.5508046 ,
          46.54734549],
        [ 64.85550796,  28.34430359,  67.42905171,  68.53148666,
          74.94972766],
        [ 56.87653028,  33.65114837,  58.91517667,  41.69785494,
          43.08453978],
        [ 87.64690692, 100.        , 100.        ,  31.65024209,
          19.38323072],
        [ 47.11225058,  33.02558524,  32.88883085,  32.21029514,
          38.46726972],
        [ 45.65905516,  41.50332988,  35.61794752,  47.43820065,
          49.03431952],
        [100.        ,  97.85992769, 100.        ,  39.0714977 ,
         100.        ],
        [ 59.092658  ,  46.99865869,  42.25355065,  49.11902626,
          46.0329485 ],
        [  0.        ,   0.        ,   0.        ,   0.        ,
           0.        ]])}

If you want to generate the profile you need to generate a similar file
"""


colors = ['moccasin','aquamarine','#dbc2ec','orchid','lightskyblue','pink','orange']
xlabels = ['A2C','PPO','SAC','SARL','DeepTrader',"AlphaMix+"]
color_idxs = [0, 1,2,3,4,5,6]
ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))
from scipy.stats.stats import find_repeats
xlabel=r'total return score $(\tau)$',
dict=tt_dict_crypto
algorithms = ['A2C','PPO','SAC','SARL','DeepTrader',"AlphaMix+"]
def make_distribution_plot(dict,algorithms,reps,xlabel,dic,color):
  score_dict = {key: dict[key][:] for key in algorithms}
  ATARI_100K_TAU = np.linspace(-1, 100,1000)
  score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict, ATARI_100K_TAU, reps=reps)
  fig, ax = plt.subplots(ncols=1, figsize=(8.0, 4.0))
  plot_utils.plot_performance_profiles(
  score_distributions, ATARI_100K_TAU,
  performance_profile_cis=score_distributions_cis,
  colors=color,
  xlabel=xlabel,
  labelsize='xx-large',
  ax=ax)
  ax.axhline(0.5, ls='--', color='k', alpha=0.4)
  fake_patches = [mpatches.Patch(color=color[alg], 
                               alpha=0.75) for alg in algorithms]
  legend = fig.legend(fake_patches, algorithms, loc='upper center', 
                    fancybox=True, ncol=len(algorithms), 
                    fontsize='small',
                    bbox_to_anchor=(0.5, 0.9,0,0))
  plt.savefig(dic,bbox_inches = 'tight')
make_distribution_plot(dict,algorithms,2000,xlabel,"./distribution",ATARI_100K_COLOR_DICT)