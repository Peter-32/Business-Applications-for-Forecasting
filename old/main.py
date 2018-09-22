# Imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandasql import sqldf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa import stattools
from scipy.stats import mstats
q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (15.0, 5.0)

def _print_null_hypothesis__is_stationary(time_series, time_series_name):
    print("Testing {}, the null hypothesis is that the series is stationary.  If all are accepted, let's just say the data is essentially random.".format(time_series_name))
    alpha = 0.05
    acf_time_series, confint_time_series, qstat_time_series, pvalues_time_series = stattools.acf(time_series,unbiased=True,nlags=20,qstat=True,alpha=alpha)
    for l, p_val in enumerate(pvalues_time_series):
        if p_val > alpha:
            print("Null hypothesis is accepted at lag = {} for p-val = {}".format(l, p_val))
        else:
            print("Null hypothesis is rejected at lag = {} for p-val = {}".format(l, p_val))
    print("")

def plot_page_visits_and_first_order_differencing(page_visits, first_order_diff):
    fig, ax = plt.subplots(2, sharex=True)
    page_visits.plot(ax=ax[0], color='b')
    ax[0].set_title("Page visits")
    first_order_diff.plot(ax=ax[1], color='r')
    ax[1].set_title("First-order differences to page visits")
    plt.savefig('plots/1_page_visits.png')
    print("1_page_visits plot created\n")


def plot_ACF_page_visits_and_first_order_differencing(page_visits, first_order_diff):
    fig, ax = plt.subplots(2, sharex=True)
    autocorrelation_plot(page_visits, ax=ax[0], color='b')
    ax[0].set_title("ACF - Page visits")
    autocorrelation_plot(first_order_diff.iloc[1:], ax=ax[1], color='r')
    ax[1].set_title("ACF - First-order differences to page visits")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.savefig('plots/2_acf.png')
    print("2_acf plot created\n")

def print_dicky_fuller_test_results(series):
    print("Null hypothesis is that it is not stationary.")
    time_series = pd.Series(series)
    adf_result = stattools.adfuller(time_series, autolag='AIC')
    print('p-val of the ADF test in air miles flown:', adf_result[1])
    print(adf_result)


# Get a Time Series

df = pd.read_csv('data/train_1.csv', sep=",", nrows=1)
df = df.T
df.columns = df.iloc[0]
df = df.rename(index=str, columns= {'2NE1_zh.wikipedia.org_all-access_spider': 'y' })
df = df.iloc[1:]
df['y'] = df['y'].astype(str).astype(int)
df.loc[:,'y'] = mstats.winsorize(df['y'].values, limits=[0.03, 0.03])
# df.loc[:,'y'] = boxcox(df['y'], -0.5)

# Stationary the Time Series

# df['y'] = df['y'].apply(lambda x: np.log(x))
MA4 = df['y'].rolling(window=4).mean()
TwoXMA4 = MA4.rolling(window=2).mean()
TwoXMA4 = TwoXMA4.loc[~pd.isnull(TwoXMA4)]

residuals = df['y'] - TwoXMA4
residuals = residuals.loc[~pd.isnull(residuals)]




fig, ax = plt.subplots(4, sharex=True)
df['y'].plot(ax=ax[0], color='b')
ax[0].set_title("Page visits")
MA4.plot(ax=ax[1], color='r')
ax[1].set_title("MA times series")
TwoXMA4.plot(ax=ax[2], color='g')
ax[2].set_title("TwoXMA4 times series")
residuals.plot(ax=ax[3], color='y')
ax[3].set_title("residuals times series")
plt.savefig('plots/3_MA.png')
print("3_MA plot created\n")


_print_null_hypothesis__is_stationary(residuals, "residuals")
plot_page_visits_and_first_order_differencing(df, residuals)
plot_ACF_page_visits_and_first_order_differencing(df, residuals)


print_dicky_fuller_test_results(df['y'])
