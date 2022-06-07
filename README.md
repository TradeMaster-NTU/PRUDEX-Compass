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
under the `portfolio_management/sunrise-pm` we can get the the name of our dataset under which lies the the name of model indicating its super-parameters under which lies the result for different seed which contains the overall result on the valid and test dataset in `result.csv` and more specifically, action and daily return for each epoch on the test set in `test_daily_action_(number of epoch).npy` and `test_daily_return_(number of epoch).csv`.

For user to pick the best model, we also add a `sz50.ipynb` in the example which help pick the best model, calculate a series of financial indicators, and summrize the result.

## Compass
The `PRUDEX-Compass` provides support for 
- A systematic evaluation from 6 axes 
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/Final%20compass.png" width = 200 height = 200 />
</div>

- A octagon to evaluate profitability,risk-control and diversity
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/octagon.PNG" width = 400 height = 200 />
</div>

- A graph discribing the dirstribution of the score of different algorithms
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/overall.png" width = 300 height = 150 />
</div>

- A graph discribing the rank informatino for different algorithms
<div align="center">
  <img src="https://github.com/qinmoelei/PRUDEX-Compass/blob/main/Compass/pictures/rank.png" width = 300 height = 150 />
</div>