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
import sys

sys.path.append(".")
"""
here we use the crypto data as an example, yet the overall data is too huge to upload.
The key point is to build a dictionary with the name of the algorithms as a key and 2-dimensional array as its contens
(1d for seed and 1d for task, in the following contents, the task is the multi-time rolling)
Alert: the location for the file is not uniformed so you have to replace it with your own files' position
"""

import seaborn as sns

tt_dict_crypto = np.load('Compass/generate/distribution/dict.npy',
                         allow_pickle='TRUE')
tt_dict_crypto = tt_dict_crypto.item()
colors = sns.color_palette("Accent")

colors = [
    'moccasin', 'aquamarine', '#dbc2ec', 'orchid', 'lightskyblue', 'pink',
    'orange'
]
xlabels = ['A2C', 'PPO', 'SAC', 'SARL', 'DeepTrader', "AlphaMix+"]
color_idxs = [0, 1, 2, 3, 4, 5, 6]
ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))
from scipy.stats.stats import find_repeats

xlabel = r'total return score $(\tau)$',
dict = tt_dict_crypto
algorithms = ['A2C', 'PPO', 'SAC', 'SARL', 'DeepTrader', "AlphaMix+"]


def make_distribution_plot(dict, algorithms, reps, xlabel, dic, color):
    score_dict = {key: dict[key][:] for key in algorithms}
    ATARI_100K_TAU = np.linspace(-1, 100, 1000)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, ATARI_100K_TAU, reps=reps)
    fig, ax = plt.subplots(ncols=1, figsize=(8.0, 4.0))
    plot_utils.plot_performance_profiles(
        score_distributions,
        ATARI_100K_TAU,
        performance_profile_cis=score_distributions_cis,
        colors=color,
        xlabel=xlabel,
        labelsize='xx-large',
        ax=ax)
    ax.axhline(0.5, ls='--', color='k', alpha=0.4)
    fake_patches = [
        mpatches.Patch(color=color[alg], alpha=0.75) for alg in algorithms
    ]
    legend = fig.legend(fake_patches,
                        algorithms,
                        loc='upper center',
                        fancybox=True,
                        ncol=len(algorithms),
                        fontsize='small',
                        bbox_to_anchor=(0.5, 0.9, 0, 0))
    plt.savefig(dic, bbox_inches='tight')


make_distribution_plot(dict, algorithms, 2000, xlabel, "./distribution",
                       ATARI_100K_COLOR_DICT)
