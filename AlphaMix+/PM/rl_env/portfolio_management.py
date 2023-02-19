import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
# from finrl.apps import config
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.finrl_meta.preprocessor.preprocessors import data_split

class TrainEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 config,
                 lookback=252,
                 turbulence_threshold=False,
                 day=0):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.lookback = lookback
        self.df = config["df"]
        self.stock_dim = config["stock_dim"]
        self.hmax = config["hmax"]
        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.reward_scaling = config["reward_scaling"]
        self.state_space = config["state_space"]
        self.action_space = config["action_space"]+1

        self.tech_indicator_list = config["tech_indicator_list"]
        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.tech_indicator_list), self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        # self.covs = self.data['cov_list'].values[0]
        self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
        self.state = self.state.reshape(1, -1)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)

        if self.terminal:
            # df = pd.DataFrame(self.portfolio_return_memory)
            # df.columns = ['daily_return']
            # plt.plot(df.daily_return.cumsum(),'r')
            # plt.savefig('results/cumulative_reward.png')
            # plt.close()

            # plt.plot(self.portfolio_return_memory,'r')
            # plt.savefig('results/rewards.png')
            # plt.close()

            # print("Done!")

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))
            tr = self.portfolio_value/self.asset_memory[0] - 1
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_daily_return['daily_return'].mean() / \
                         df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, [tr, sharpe, df_daily_return['daily_return']]
        else:
            # print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            # if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            # else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # self.covs = self.data['cov_list'].values[0]
            # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.state = np.array([self.data[tech].values.tolist() for tech in self.tech_indicator_list])
            self.state = np.array(self.state)
            portfolio_weights = weights[:, 1:]
            # print(self.state)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = np.sum(((self.data.close.values / last_day_memory.close.values) - 1) * portfolio_weights)

            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.reward = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            # self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            self.state = self.state.reshape(1, -1)
            self.reward = self.reward * self.reward_scaling
            # print("各个股票收益",((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # print("return",portfolio_return)
            # print(self.portfolio_value)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        # self.covs = self.data['cov_list'].values[0]
        self.state = [self.data[tech].values.tolist() for tech in self.tech_indicator_list]
        self.state = np.array(self.state)
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.state = self.state.reshape(1, -1)
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

def clean_data(data):
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
    return df

def retrivedata_from_csv(file_dir):
    import os
    all_file_list=os.listdir(file_dir)
    for single_file in all_file_list:
        single_data_frame=pd.read_csv(os.path.join(file_dir,single_file))
        if os.path.splitext(single_file)[1] == '.csv':
            tic = os.path.splitext(single_file)[0]
            single_data_frame["tic"]=tic
        if single_file ==all_file_list[0]:
            all_data_frame=single_data_frame
        else:  #进行concat操作       
            all_data_frame=pd.concat([all_data_frame,single_data_frame],ignore_index=False)
    return all_data_frame

def clean_acl18data(data):
    """A little data preprocessing for acl18data only so that it is in the form of Yahoo_data """
    data.drop(columns=["Volume"],inplace=True)
    data=data.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjcp",})
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    return df

def clean_sz50data(data):
    """A little data preprocessing for acl18data only so that it is in the form of Yahoo_data """
    data.drop(columns=["Unnamed: 0","Date","Volume","Vwap","norm_open","norm_high","norm_low","norm_close","norm_vwap","Amount","Symbol"],inplace=True)
    data=data.rename(columns={"Time":"date","Open":"open","High":"high","Low":"low","Close":"close"})
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    df["adjcp"]=df.close
    return df


def clean_crypto(data):
    """A little data preprocessing for crypto only so that it is in the form of Yahoo_data """
    data.drop(columns=["SNo","Marketcap","tic","Volume","Name"],inplace=True)
    df=data.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Symbol":"tic",})
    all_tic_list=df.tic.unique()
    good_tic=[]
    for tic in all_tic_list:
        if min(df[df.tic==tic].date.unique())<='2016-01-01 23:59:59':
            good_tic.append(tic)
    df_good=pd.DataFrame()
    for tic in good_tic:
        df_good_single=df[df.tic==tic]
        df_good=pd.concat([df_good,df_good_single],ignore_index=False)
    df=df_good[df_good.date>='2016-01-01 23:59:59']
    def clean_data(data):
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        return df
    df=clean_data(df)
    df["adjcp"]=df["close"]
    return df


def clean_Futures(data):
    data.drop(columns=["Volume"],inplace=True)
    df=data.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adjcp",})
    df=df[df.date>='2008-01-01']
    df = df.copy()
    df = df.sort_values(["tic", "date"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)]
    return df




def preprocess(df):
    df["zopen"]=df["open"]/df["close"]-1
    df["zhigh"]=df["high"]/df["close"]-1
    df["zlow"]=df["low"]/df["close"]-1
    df["zadjcp"]=df["adjcp"]/df["close"]-1
    df_new=df.sort_values(by=["tic", "date"])
    stock = df_new
    unique_ticker = df_new.tic.unique()
    df_indicator=pd.DataFrame()
    for i in range(len(unique_ticker)):
        temp_indicator = stock[stock.tic == unique_ticker[i]]
        temp_indicator["zclose"]=(temp_indicator.close/(temp_indicator.close.rolling(2).sum()-temp_indicator.close))-1
        temp_indicator["zd_5"]=(temp_indicator.adjcp.rolling(5).sum()/5)/temp_indicator.adjcp-1
        temp_indicator["zd_10"]=(temp_indicator.adjcp.rolling(10).sum()/10)/temp_indicator.adjcp-1
        temp_indicator["zd_15"]=(temp_indicator.adjcp.rolling(15).sum()/15)/temp_indicator.adjcp-1
        temp_indicator["zd_20"]=(temp_indicator.adjcp.rolling(20).sum()/20)/temp_indicator.adjcp-1
        temp_indicator["zd_25"]=(temp_indicator.adjcp.rolling(25).sum()/25)/temp_indicator.adjcp-1
        temp_indicator["zd_30"]=(temp_indicator.adjcp.rolling(30).sum()/30)/temp_indicator.adjcp-1
        df_indicator=df_indicator.append(temp_indicator,ignore_index=True)
    df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
    return df_indicator

def get_dataset_config(dataset):





    if dataset == 'dj30':
        dp = YahooFinanceProcessor()
        df = dp.download_data(start_date='2012-01-01',
                              end_date='2022-01-01',
                              ticker_list=[
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW"
], time_interval='1D')

        df.drop(columns="volume", inplace=True)
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            merged_closes = df.pivot_table(index="date", columns="tic", values="close")
            merged_closes = merged_closes.dropna(axis=1)
            tics = merged_closes.columns
            df = df[df.tic.isin(tics)]
            return df
        
        df = clean_data(df)
        df["zopen"] = df["open"] / df["close"] - 1
        df["zhigh"] = df["high"] / df["close"] - 1
        df["zlow"] = df["low"] / df["close"] - 1
        df["zadjcp"] = df["adjcp"] / df["close"] - 1
        df_new = df.sort_values(by=["tic", "date"])
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"] = (temp_indicator.close / (
                        temp_indicator.close.rolling(2).sum() - temp_indicator.close)) - 1
            temp_indicator["zd_5"] = (temp_indicator.adjcp.rolling(5).sum() / 5) / temp_indicator.adjcp - 1
            temp_indicator["zd_10"] = (temp_indicator.adjcp.rolling(10).sum() / 10) / temp_indicator.adjcp - 1
            temp_indicator["zd_15"] = (temp_indicator.adjcp.rolling(15).sum() / 15) / temp_indicator.adjcp - 1
            temp_indicator["zd_20"] = (temp_indicator.adjcp.rolling(20).sum() / 20) / temp_indicator.adjcp - 1
            temp_indicator["zd_25"] = (temp_indicator.adjcp.rolling(25).sum() / 25) / temp_indicator.adjcp - 1
            temp_indicator["zd_30"] = (temp_indicator.adjcp.rolling(30).sum() / 30) / temp_indicator.adjcp - 1
            df_indicator = df_indicator.append(temp_indicator, ignore_index=True)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2012-01-01', '2020-01-01')
        valid = data_split(df_indicator, '2020-01-01', '2021-01-01')
        test = data_split(df_indicator, '2021-01-01', '2022-01-01')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension









    if dataset == 'dj30_rolling1':
        dp = YahooFinanceProcessor()
        df = dp.download_data(start_date='2012-01-01',
                              end_date='2022-01-01',
                              ticker_list=[
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW"
], time_interval='1D')

        df.drop(columns="volume", inplace=True)
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            merged_closes = df.pivot_table(index="date", columns="tic", values="close")
            merged_closes = merged_closes.dropna(axis=1)
            tics = merged_closes.columns
            df = df[df.tic.isin(tics)]
            return df
        df = clean_data(df)
        df["zopen"] = df["open"] / df["close"] - 1
        df["zhigh"] = df["high"] / df["close"] - 1
        df["zlow"] = df["low"] / df["close"] - 1
        df["zadjcp"] = df["adjcp"] / df["close"] - 1
        df_new = df.sort_values(by=["tic", "date"])
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"] = (temp_indicator.close / (
                        temp_indicator.close.rolling(2).sum() - temp_indicator.close)) - 1
            temp_indicator["zd_5"] = (temp_indicator.adjcp.rolling(5).sum() / 5) / temp_indicator.adjcp - 1
            temp_indicator["zd_10"] = (temp_indicator.adjcp.rolling(10).sum() / 10) / temp_indicator.adjcp - 1
            temp_indicator["zd_15"] = (temp_indicator.adjcp.rolling(15).sum() / 15) / temp_indicator.adjcp - 1
            temp_indicator["zd_20"] = (temp_indicator.adjcp.rolling(20).sum() / 20) / temp_indicator.adjcp - 1
            temp_indicator["zd_25"] = (temp_indicator.adjcp.rolling(25).sum() / 25) / temp_indicator.adjcp - 1
            temp_indicator["zd_30"] = (temp_indicator.adjcp.rolling(30).sum() / 30) / temp_indicator.adjcp - 1
            df_indicator = df_indicator.append(temp_indicator, ignore_index=True)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2012-01-01','2019-01-01')
        valid= data_split(df_indicator, '2019-01-01','2020-01-01')
        test=data_split(df_indicator,'2020-01-01','2021-01-01')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension






    if dataset == 'dj30_rolling2':
        dp = YahooFinanceProcessor()
        df = dp.download_data(start_date='2012-01-01',
                              end_date='2022-01-01',
                              ticker_list=[
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW"
], time_interval='1D')

        df.drop(columns="volume", inplace=True)
        
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            merged_closes = df.pivot_table(index="date", columns="tic", values="close")
            merged_closes = merged_closes.dropna(axis=1)
            tics = merged_closes.columns
            df = df[df.tic.isin(tics)]
            return df
        df = clean_data(df)
        df["zopen"] = df["open"] / df["close"] - 1
        df["zhigh"] = df["high"] / df["close"] - 1
        df["zlow"] = df["low"] / df["close"] - 1
        df["zadjcp"] = df["adjcp"] / df["close"] - 1
        df_new = df.sort_values(by=["tic", "date"])
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"] = (temp_indicator.close / (
                        temp_indicator.close.rolling(2).sum() - temp_indicator.close)) - 1
            temp_indicator["zd_5"] = (temp_indicator.adjcp.rolling(5).sum() / 5) / temp_indicator.adjcp - 1
            temp_indicator["zd_10"] = (temp_indicator.adjcp.rolling(10).sum() / 10) / temp_indicator.adjcp - 1
            temp_indicator["zd_15"] = (temp_indicator.adjcp.rolling(15).sum() / 15) / temp_indicator.adjcp - 1
            temp_indicator["zd_20"] = (temp_indicator.adjcp.rolling(20).sum() / 20) / temp_indicator.adjcp - 1
            temp_indicator["zd_25"] = (temp_indicator.adjcp.rolling(25).sum() / 25) / temp_indicator.adjcp - 1
            temp_indicator["zd_30"] = (temp_indicator.adjcp.rolling(30).sum() / 30) / temp_indicator.adjcp - 1
            df_indicator = df_indicator.append(temp_indicator, ignore_index=True)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2012-01-01','2018-01-01')
        valid= data_split(df_indicator, '2018-01-01','2019-01-01')
        test=data_split(df_indicator,'2019-01-01','2020-01-01')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension











    if dataset=='sz50':
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/sz50/")#the location where the sz50 data is
        df=clean_sz50data(df)
        df=preprocess(df)
        train = data_split(df, '2012-09-04','2019-01-01')
        valid= data_split(df, '2019-01-01','2020-01-01')
        test=data_split(df,'2020-01-01', '2020-08-31')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension


    if dataset=='sz50_rolling1':
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/sz50/")#the location where the sz50 data is
        df=clean_sz50data(df)
        df=preprocess(df)
        train = data_split(df, '2012-09-04','2018-01-01')
        valid= data_split(df, '2018-01-01','2019-01-01')
        test=data_split(df,'2019-01-01', '2019-08-28')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension

    
    if dataset=='sz50_rolling2':
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/sz50/")#the location where the sz50 data is
        df=clean_sz50data(df)
        df=preprocess(df)
        train = data_split(df, '2012-09-04','2017-01-01')
        valid= data_split(df, '2017-01-01','2018-01-01')
        test=data_split(df,'2018-01-01', '2018-08-28')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension












    if dataset=="crypto":
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/crypto")
        df=clean_crypto(df)
        df=preprocess(df)
        train = data_split(df, '2016-01-01','2020-01-01')
        valid= data_split(df, '2020-01-01','2021-01-01')
        test=data_split(df,'2021-01-01', '2021-07-06')
        tech_indicator_list=["zopen","zhigh","zlow","zadjcp","zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension
    
    if dataset=="crypto_rolling1":
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/crypto")
        df=clean_crypto(df)
        df=preprocess(df)
        train = data_split(df, '2016-01-01','2019-01-01')
        valid= data_split(df, '2019-01-01','2020-01-01')
        test=data_split(df,'2020-01-01', '2020-07-06')
        tech_indicator_list=["zopen","zhigh","zlow","zadjcp","zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension


    if dataset=="crypto_rolling2":
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/crypto")
        df=clean_crypto(df)
        df=preprocess(df)
        train = data_split(df, '2016-01-01','2018-01-01')
        valid= data_split(df, '2018-01-01','2019-01-01')
        test=data_split(df,'2019-01-01', '2019-07-06')
        tech_indicator_list=["zopen","zhigh","zlow","zadjcp","zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension






















    if dataset=="foreign exchange":
        """notice that the form of the foreign exchange is a little different  becasue it only gets close price instead o c a h l like others"""
        df=pd.read_csv("/home/sunshuo/qml/RL_Mix/PM/data/foreign_exchange/Foreign_Exchange_Rates.csv")
        df=df[df.columns[1:]]
        df=df.replace("ND",np.nan)
        df=df.dropna(axis=0,how='any')
        dollar_name=df.columns[1:]
        datetime=df["Time Serie"]
        df=df[dollar_name]
        for i in range(5015):
            for j in range(22):
                df.iloc[i,j]=1/float(df.iloc[i,j])
        tic_names=df.columns
        new_tic_names=[]
        for tic_name in tic_names:
            start_position=tic_name.find('-')
            end_position=tic_name.find('/')
            tic_name=tic_name[start_position+2:end_position]
            new_tic_names.append(tic_name)
        df.columns=new_tic_names
        single_tic=pd.DataFrame(df[tic_name])
        single_tic["tic"]=tic_name
        single_tic.columns=["price","tic"]
        df_new=pd.DataFrame()
        for tic_name in new_tic_names:
            single_tic=pd.DataFrame(df[tic_name])
            single_tic["tic"]=tic_name
            single_tic.columns=["price","tic"]
            new_df=pd.concat([datetime,single_tic],axis=1)
            df_new=pd.concat([df_new,
                    new_df],ignore_index=False)
        df_new.columns=["date","close","tic"]
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            return df
        df_new=clean_data(df_new)
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator=pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"]=(temp_indicator.close/(temp_indicator.close.rolling(2).sum()-temp_indicator.close))-1
            temp_indicator["zd_5"]=(temp_indicator.close.rolling(5).sum()/5)/temp_indicator.close-1
            temp_indicator["zd_10"]=(temp_indicator.close.rolling(10).sum()/10)/temp_indicator.close-1
            temp_indicator["zd_15"]=(temp_indicator.close.rolling(15).sum()/15)/temp_indicator.close-1
            temp_indicator["zd_20"]=(temp_indicator.close.rolling(20).sum()/20)/temp_indicator.close-1
            temp_indicator["zd_25"]=(temp_indicator.close.rolling(25).sum()/25)/temp_indicator.close-1
            temp_indicator["zd_30"]=(temp_indicator.close.rolling(30).sum()/30)/temp_indicator.close-1
            df_indicator=df_indicator.append(temp_indicator,ignore_index=False)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2000-01-01','2018-01-01')
        valid= data_split(df_indicator, '2018-01-01','2019-01-01')
        test=data_split(df_indicator,'2019-01-01', '2020-01-01')
        tech_indicator_list=["zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension







    if dataset=="foreign exchange_rolling1":
        """notice that the form of the foreign exchange is a little different  becasue it only gets close price instead o c a h l like others"""
        df=pd.read_csv("/home/sunshuo/qml/RL_Mix/PM/data/foreign_exchange/Foreign_Exchange_Rates.csv")
        df=df[df.columns[1:]]
        df=df.replace("ND",np.nan)
        df=df.dropna(axis=0,how='any')
        dollar_name=df.columns[1:]
        datetime=df["Time Serie"]
        df=df[dollar_name]
        for i in range(5015):
            for j in range(22):
                df.iloc[i,j]=1/float(df.iloc[i,j])
        tic_names=df.columns
        new_tic_names=[]
        for tic_name in tic_names:
            start_position=tic_name.find('-')
            end_position=tic_name.find('/')
            tic_name=tic_name[start_position+2:end_position]
            new_tic_names.append(tic_name)
        df.columns=new_tic_names
        single_tic=pd.DataFrame(df[tic_name])
        single_tic["tic"]=tic_name
        single_tic.columns=["price","tic"]
        df_new=pd.DataFrame()
        for tic_name in new_tic_names:
            single_tic=pd.DataFrame(df[tic_name])
            single_tic["tic"]=tic_name
            single_tic.columns=["price","tic"]
            new_df=pd.concat([datetime,single_tic],axis=1)
            df_new=pd.concat([df_new,
                    new_df],ignore_index=False)
        df_new.columns=["date","close","tic"]
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            return df
        df_new=clean_data(df_new)
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator=pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"]=(temp_indicator.close/(temp_indicator.close.rolling(2).sum()-temp_indicator.close))-1
            temp_indicator["zd_5"]=(temp_indicator.close.rolling(5).sum()/5)/temp_indicator.close-1
            temp_indicator["zd_10"]=(temp_indicator.close.rolling(10).sum()/10)/temp_indicator.close-1
            temp_indicator["zd_15"]=(temp_indicator.close.rolling(15).sum()/15)/temp_indicator.close-1
            temp_indicator["zd_20"]=(temp_indicator.close.rolling(20).sum()/20)/temp_indicator.close-1
            temp_indicator["zd_25"]=(temp_indicator.close.rolling(25).sum()/25)/temp_indicator.close-1
            temp_indicator["zd_30"]=(temp_indicator.close.rolling(30).sum()/30)/temp_indicator.close-1
            df_indicator=df_indicator.append(temp_indicator,ignore_index=False)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2000-01-01','2017-01-01')
        valid= data_split(df_indicator, '2017-01-01','2018-01-01')
        test=data_split(df_indicator,'2018-01-01', '2019-01-01')
        tech_indicator_list=["zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension





    if dataset=="foreign exchange_rolling2":
        """notice that the form of the foreign exchange is a little different  becasue it only gets close price instead o c a h l like others"""
        df=pd.read_csv("/home/sunshuo/qml/RL_Mix/PM/data/foreign_exchange/Foreign_Exchange_Rates.csv")
        df=df[df.columns[1:]]
        df=df.replace("ND",np.nan)
        df=df.dropna(axis=0,how='any')
        dollar_name=df.columns[1:]
        datetime=df["Time Serie"]
        df=df[dollar_name]
        for i in range(5015):
            for j in range(22):
                df.iloc[i,j]=1/float(df.iloc[i,j])
        tic_names=df.columns
        new_tic_names=[]
        for tic_name in tic_names:
            start_position=tic_name.find('-')
            end_position=tic_name.find('/')
            tic_name=tic_name[start_position+2:end_position]
            new_tic_names.append(tic_name)
        df.columns=new_tic_names
        single_tic=pd.DataFrame(df[tic_name])
        single_tic["tic"]=tic_name
        single_tic.columns=["price","tic"]
        df_new=pd.DataFrame()
        for tic_name in new_tic_names:
            single_tic=pd.DataFrame(df[tic_name])
            single_tic["tic"]=tic_name
            single_tic.columns=["price","tic"]
            new_df=pd.concat([datetime,single_tic],axis=1)
            df_new=pd.concat([df_new,
                    new_df],ignore_index=False)
        df_new.columns=["date","close","tic"]
        def clean_data(data):
            df = data.copy()
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]
            return df
        df_new=clean_data(df_new)
        stock = df_new
        unique_ticker = stock.tic.unique()
        df_indicator=pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_indicator = stock[stock.tic == unique_ticker[i]]
            temp_indicator["zclose"]=(temp_indicator.close/(temp_indicator.close.rolling(2).sum()-temp_indicator.close))-1
            temp_indicator["zd_5"]=(temp_indicator.close.rolling(5).sum()/5)/temp_indicator.close-1
            temp_indicator["zd_10"]=(temp_indicator.close.rolling(10).sum()/10)/temp_indicator.close-1
            temp_indicator["zd_15"]=(temp_indicator.close.rolling(15).sum()/15)/temp_indicator.close-1
            temp_indicator["zd_20"]=(temp_indicator.close.rolling(20).sum()/20)/temp_indicator.close-1
            temp_indicator["zd_25"]=(temp_indicator.close.rolling(25).sum()/25)/temp_indicator.close-1
            temp_indicator["zd_30"]=(temp_indicator.close.rolling(30).sum()/30)/temp_indicator.close-1
            df_indicator=df_indicator.append(temp_indicator,ignore_index=False)
        df_indicator = df_indicator.fillna(method="ffill").fillna(method="bfill")
        train = data_split(df_indicator, '2000-01-01','2016-01-01')
        valid= data_split(df_indicator, '2016-01-01','2017-01-01')
        test=data_split(df_indicator,'2017-01-01', '2018-01-01')
        tech_indicator_list=["zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension












































    if dataset=="Futures":
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/Futures")
        df=clean_Futures(df)
        df=preprocess(df)
        train = data_split(df, '2008-01-01','2019-01-01')
        valid= data_split(df, '2019-01-01','2020-01-01')
        test=data_split(df,'2020-01-01', '2021-01-01')
        tech_indicator_list=["zopen","zhigh","zlow","zadjcp","zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension






    



    if dataset=='acl18':
        df=retrivedata_from_csv("/home/sunshuo/qml/RL_Mix/PM/data/acl18/")#the location where the sz50 data is
        df=clean_acl18data(df)
        df=preprocess(df)
        train = data_split(df, '2012-09-04','2015-09-01')
        valid= data_split(df, '2015-09-01','2016-09-01')
        test=data_split(df,'2016-09-01', '2017-09-01')
        tech_indicator_list = ["zopen", "zhigh", "zlow", "zadjcp", "zclose", "zd_5", "zd_10", "zd_15", "zd_20", "zd_25",
                               "zd_30"]
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension
    




    train_config = {
        "df": train,
        "hmax": 100,
        "initial_amount": 1e6,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    valid_config = {
        "df": valid,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4

    }
    test_config = {
        "df": test,
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4

    }
    return train_config, valid_config, test_config
    

if __name__ == "__main__":
    train_config, valid_config, test_config=get_dataset_config("sz50")
    print(train_config["df"].shape)
    print(valid_config["df"].shape)
    print(test_config["df"].shape)
    print(train_config["df"])
