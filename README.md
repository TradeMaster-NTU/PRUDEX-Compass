# PRUDEX-Compass
The contribution of PRUDEX-COmpass consists of 2 parts: 
- Propose AlphaMix+ as a strong FinRL baseline, which leverages Mixture-of-Experts (MoE) and risk-sensitive approaches to make diversified risk-aware investment decisions
- Introduce PRUDEX-Compass, which has 6 axes, i.e., Profitability, Risk-control, Universality, Diversity, rEliability, and eXplainability, with a total of 16 measures for a systematic evaluation

## AlphaMix+
AlphaMix+, a universal RL framework with diversified risk-aware Mixture-of-Experts(MoE) for quantitative trading.

We implement it through using and here is how you can choose your dataset and tune the parameters, train valid and test.
Here is the sctructure of the AlphaMix+.
```
-- PM
    |-- LICENSE
    |-- README.md
    |-- data
    |-- environment
    |-- finrl
    |-- mbbl
    |-- rl_env
    |-- rlkit
    |-- sac_gym.py
    |-- scripts
    |-- setup.py
    |-- sunrise_gym.py
    `-- sunrise_pm_sz50.py

```

### Example
Here we use the sz50 dataset as an example to show you how we can use it.
We can directly open the `sunrise_pm_sz50.py` under `PM`.You can dierctly run it or run it on the `bash` using 
```
python ./sunrise_pm_sz50.py --dataset sz50 --num_layer 4
```
or any paramaters you want to change which is defined in the `parse_args` function.

After traning, it will store the result under `data` whose structure is like
```
|-- crypto
|-- foreign_exchange
|-- portfolio_management
|   |-- sunrise-pm
|       |-- sz50
|       |   `-- en_3_batch_256_plr_0.0007_qlr_0.0007_layer_128_2_buffer_10000_discount_0.99_tem_20_bm_0.5_uncertain_0.5
|       |       |-- seed_12345
|       |       |   |-- debug.log
|       |       |   |-- progress.csv
|       |       |   |-- result.csv
|       |       |   |-- test_daily_action_0.npy
|       |       |   |-- test_daily_action_1.npy
|       |       |   |-- ...
|       |       |   |-- test_daily_return_0.csv
|       |       |   |-- test_daily_return_1.csv
|       |       |   |-- ...
|       |       |   `-- variant.json
|       |       |-- ...
|       |       |-- sz50.ipynb
|-- sz50

```
Under the `portfolio_management/sunrise-pm`, we can get the the name of our dataset under which lies the the name of model indicating its super-parameters under which lies the result for different seed which contains the overall result on the valid and test dataset in `result.csv` and more specifically, action and daily return for each epoch on the test set in `test_daily_action_(number of epoch).npy` and `test_daily_return_(number of epoch).csv`.

For users to pick the best model, we also add a `sz50.ipynb` in the example which help pick the best model, calculate a series of financial indicators, and summrize the result.

## Compass
The `PRUDEX-Compass` provides support for 
- A systematic evaluation from 6 axes 
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/Final%20compass.png" width = 200 height = 200 />
</div>

And here is the file structure for `Final compass`
```
|-- Finall compass
|   |-- blank.tex
|   |-- example.tex
|   |-- main.tex

```
Here we provide a blank tex that you can play with, the blank tex does not have any color block but the hexagon and the outer ring, while the example tex generate the picture shown above. we can use the main.tex to see it. You can also alter the config or the colors of the compass.

- A octagon to evaluate profitability,risk-control and diversity
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/octagon.PNG" width = 400 height = 200 />
</div>

And here is the file structure for `octagon`
```
-- ocatgon
    |-- A2C.tex
    |-- Alphamix+.tex
    |-- DeepTrader.tex
    |-- PPO.tex
    |-- SAC.tex
    |-- SARL.tex
    |-- blank.tex
```
Here we provide a blank tex that you can play with, the blank tex does not have any color block but the hexagon and the outer ring, while the rest of tex generate the subpicture corresponding to the  shown above. You can also manipulate the color and the value for different algorithms to generate graphs.

- A graph discribing the dirstribution of the score of different algorithms
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/overall.png" width = 300 height = 150 />
</div>

The key is to generate a dictionary whose key is the name of algorithms and the value is 2d array which represents different seeds and different task, then with the dictionary naming `overall_dict`, we can simpily use the code
```
colors = ['moccasin','aquamarine','#dbc2ec','orchid','lightskyblue','pink','orange']
xlabels = ['A2C','PPO','SAC','SARL','DeepTrader',"AlphaMix+"]
color_idxs = [0, 1,2,3,4,5,6]
ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))
from scipy.stats.stats import find_repeats
#@title Calculate score distributions and average score distributions for for Atari 100k

algorithms = ['A2C','PPO','SAC','SARL','DeepTrader',"AlphaMix+"]

score_dict = {key: overall_dict[key][:] for key in algorithms}
ATARI_100K_TAU = np.linspace(-1, 100,1000)
# Higher value of reps corresponds to more accurate estimates but are slower
# to computed. `reps` corresponds to number of bootstrap resamples.
reps = 2000

score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict, ATARI_100K_TAU, reps=reps)

fig, ax = plt.subplots(ncols=1, figsize=(8.0, 4.0))

plot_utils.plot_performance_profiles(
  score_distributions, ATARI_100K_TAU,
  performance_profile_cis=score_distributions_cis,
  colors=ATARI_100K_COLOR_DICT,
  xlabel=r'total return score $(\tau)$',
  labelsize='xx-large',
  ax=ax)

ax.axhline(0.5, ls='--', color='k', alpha=0.4)
fake_patches = [mpatches.Patch(color=ATARI_100K_COLOR_DICT[alg], 
                               alpha=0.75) for alg in algorithms]
legend = fig.legend(fake_patches, algorithms, loc='upper center', 
                    fancybox=True, ncol=len(algorithms), 
                    fontsize='small',
                    bbox_to_anchor=(0.5, 0.9,0,0))
plt.savefig("./distribution.pdf",bbox_inches = 'tight')
```
to generate the distribution. Notice that we only use one dicator (total return in the example) to demonstrate the graph, which is a little different from what we have next(rank information).

For more precise informatino, please refer to `Compass/generate/distribution/distribution.py`

- A graph discribing the rank informatino for different algorithms
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/rank.png" width = 300 height = 150 />
</div> 

The key is to generate a dictionary whose key is the name of indicator and the value is dictionary similar to what we have in the distribution. Then we can simpliy use the code
```
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
    # Sort based on negative scores as rank 0 corresponds to minimum value,
    # rank 1 corresponds to second minimum value when using lexsort.
    task_x = -X[:, :, task]
    # This is done to randomly break ties.
    rand_x = np.random.random(size=task_x.shape)
    # Last key is the primary key, 
    indices = np.lexsort((rand_x, task_x), axis=0)
    mat = np.zeros((num_algs, num_algs))
    for rank in range(num_algs):
      cnts = collections.Counter(indices[rank])
      mat[:, rank] = np.array([cnts[i]/n for i in range(num_algs)])
    all_mat.append(mat)
  all_mat = np.stack(all_mat, axis=0)
  return all_mat
mean_ranks_all = {}
all_ranks_individual = {}
for key in ['tr','sr','sor','cr']:
  dmc_score_dict = dmc_scores[key]
  algs =  ['A2C','PPO','SAC','SARL','DeepTrader','AlphaMix+']
  all_ranks = get_rank_matrix(dmc_score_dict, 200000, algorithms=algs)
  mean_ranks_all[key] = np.mean(all_ranks, axis=0)
  all_ranks_individual[key] = all_ranks
colors =['moccasin','aquamarine','#dbc2ec','salmon','lightskyblue','pink','orange']

algs = ['A2C','PPO','SAC','SARL','DeepTrader',"AlphaMix+"]
color_idxs = [0, 1,2,3,4,5,6]
DMC_COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))
#@title Plot aggregate ranks

keys = algs
labels = list(range(1, len(keys)+1))
width = 1.0       # the width of the bars: can also be len(x) sequence

# fig, axes = plt.subplots(ncols=2, figsize=(2.9 * 2, 3.6))
fig, axes = plt.subplots(nrows=1,ncols=4, figsize=(8, 2 * 2))



for main_idx, main_key in enumerate(['tr','sr','sor','cr']):
  # print(main_idx)
  ax = axes[main_idx]
  mean_ranks = mean_ranks_all[main_key]
  # print(mean_ranks_all)
  bottom = np.zeros_like(mean_ranks[0])
  for i, key in enumerate(algs):
    label = key if main_idx == 0 else None
    # print(label)
    ax.bar(labels, mean_ranks[i], width, label=label, 
          color=DMC_COLOR_DICT[key], bottom=bottom, alpha=0.9)
    bottom += mean_ranks[i]

  yticks = np.array(range(0, 101, 20))
  ax.set_yticklabels(yticks, size='large')
  # if main_idx == 0:
  #   ax.set_ylabel('Fraction (in %)', size='x-large')
  #   yticks = np.array(range(0, 101, 20))
  #   ax.set_yticklabels(yticks, size='large')
  # else:
  #   ax.set_yticklabels([])
  # if main_idx==0:
  #   ax.set_yticks(yticks * 0.01)
  # ax.set_xlabel('Ranking', size='x-large')
  if main_idx in [0,1,2,3,4]:
    ax.set_xticks(labels)
  else:
    ax.set_xticks([])
  #ax.set_xticklabels(labels, size='large')
  ax.set_title(main_key, size='x-large', y=0.95)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  # left = True if main_idx == 0 else False
  left= True
  ax.tick_params(axis='both', which='both', bottom=False, top=False,
                  left=left, right=False, labeltop=False,
                  labelbottom=True, labelleft=left, labelright=False)

fig.legend(loc='center right', fancybox=True, ncol=1, fontsize='large', bbox_to_anchor=(1.15, 0.35))
fig.subplots_adjust(top=0.72, wspace=0.5, bottom=0)
fig.text(x=-0.01, y=0.2, s='Fraction (in %)', rotation=90, size='xx-large')
# plt.show()
plt.savefig("./rank.pdf",bbox_inches = 'tight')
```
to generate the graph.

For more information, please refer to `rank.py`