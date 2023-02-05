from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import statsmodels.api as sm
register_matplotlib_converters()
from scipy import stats
from statsmodels.graphics.api import qqplot
import numpy as np


def delete_null_values(df):
    df.dropna(how='any',axis=0)
    return df


def create_data(df):
    df2 = df.groupby(df['time']).tail(1)
    df2 = df2.reset_index()
    df3 = df2[['time', 'dev_acc_d']]
    df2 = df2[['time', 'dev_acc_d']]
    last_index = 0
    for i in range(len(df2)):
        if last_index != i:
            diff = df2.at[i, "dev_acc_d"] - df2.at[last_index, "dev_acc_d"]
            df3.at[i, "dev_acc_d"] = diff
        last_index = i
    return df3


def find_optimal_parameters_with_pmdarima(df):
    stepwise_fit = auto_arima(df['dev_acc_d'], start_p=1, start_q=1,
                              max_p=3, max_q=3, m=5,
                              start_P=0, seasonal=False,
                              d=None, D=1, trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
    stepwise_fit.summary()


def calculate_afc_pacf(df):
    df2 = df['dev_acc_d']
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df2.values.squeeze(), lags=9, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df2, lags=4, ax=ax2)
    plt.show()


def arima(df2):
    # Create model
    model = ARIMA(df2["dev_acc_d"], order=(1, 1, 0)).fit()
    print(model.params)
    print(model.aic, model.bic, model.hqic)
    # draw resid and qqplot
    sm.stats.durbin_watson(model.resid.values)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax = model.resid.plot(ax=ax)
    resid = model.resid
    stats.normaltest(resid)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line="q", ax=ax, fit=True)
    plt.show()
    # print lags' data
    r, q, p = sm.tsa.acf(resid.values.squeeze(), fft=True, qstat=True)
    data = np.c_[np.arange(1, 10), r[1:], q, p]
    table = pd.DataFrame(data, columns=["lag", "AC", "Q", "Prob(>Q)"])
    print(table.set_index("lag"))


def main():
    df = pd.read_csv('./Data/ARIMA-dataset.csv')
    df = df.dropna(how='any', axis=0)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df2 = df[['time', 'dev_acc_d']]
    df2 = df2.reset_index()
    df2 = df2[['time', 'dev_acc_d']]
    df2 = create_data(df2)
    calculate_afc_pacf(df2)
    arima(df2)


if __name__ == '__main__':
    main()

