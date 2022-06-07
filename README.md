# PRUDEX-Compass
The contribution of PRUDEX-COmpass consists of 2 parts: 
- Propose AlphaMix+ as a strong FinRL baseline, which leverages Mixture-of-Experts (MoE) and risk-sensitive approaches to make diversified risk-aware investment decisions
- Introduce PRUDEX-Compass, which has 6 axes, i.e., Profitability, Risk-control, Universality, Diversity, rEliability, and eXplainability, with a total of 16 measures for a systematic evaluation

## AlphaMix+
AlphaMix+, a universal RL framework with diversified risk-aware Mixture-of-Experts(MoE) for quantitative trading.

We implement it through using and here is how you can choose your dataset and tune the parameters, train valid and test.
Here is the sctructure of the AlphaMix+.
```

```

### Example
Here we use the sz50 dataset as an example to show you how we can use it.



We can directly open the `sunrise_pm_sz50.py` under `PM`. 


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