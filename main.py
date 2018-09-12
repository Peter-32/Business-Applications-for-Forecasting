# Imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandasql import sqldf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa import stattools
q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (15.0, 5.0)

# Prepare data

def prepare_data():
    page_visits = pd.read_csv('data/train_1.csv', sep=",", nrows=1)
    page_visits = _transpose_and_set_column_names(page_visits)
    page_visits = _set_index_datatype_to_datetime(page_visits)
    return _convert_all_columns_to_integers(page_visits)

def _transpose_and_set_column_names(page_visits):
    page_visits = page_visits.T
    page_visits.columns = page_visits.iloc[0]
    return page_visits.iloc[1:]

def _set_index_datatype_to_datetime(page_visits):
    page_visits['_Date_'] = pd.to_datetime(page_visits.index)
    page_visits.set_index(['_Date_'])
    return page_visits.drop(['_Date_'], axis=1)

def _convert_all_columns_to_integers(page_visits):
    for col in page_visits.columns:
        page_visits[col] = page_visits[col].astype(str).astype(int)
    return page_visits

def _UNUSED_remove_missing_values_and_print_number_removed(time_series):
    missing = pd.isnull(time_series)
    print('Number of missing values found:', missing.sum())
    time_series = time_series.loc[~missing, :]


page_visits = prepare_data()

# Stationary Process (stationary time series are not dependent on time)

def first_order_differencing_checks():
    first_order_diff = _get_first_order_differencing(page_visits)
    _plot_page_visits_and_first_order_differencing(page_visits, first_order_diff)
    _plot_ACF_page_visits_and_first_order_differencing(page_visits, first_order_diff)
    _print_null_hypothesis__is_stationary(page_visits, "page_visits")
    _print_null_hypothesis__is_stationary(first_order_diff, "first_order_diff")

def _get_first_order_differencing(page_visits):
    first_order_diff = page_visits.diff(1)
    return first_order_diff.iloc[1:]

def _plot_page_visits_and_first_order_differencing(page_visits, first_order_diff):
    fig, ax = plt.subplots(2, sharex=True)
    page_visits.plot(ax=ax[0], color='b')
    ax[0].set_title("Page visits")
    first_order_diff.plot(ax=ax[1], color='r')
    ax[1].set_title("First-order differences to page visits")
    plt.savefig('plots/1_page_visits.png')
    print("1_page_visits plot created\n")

def _plot_ACF_page_visits_and_first_order_differencing(page_visits, first_order_diff):
    fig, ax = plt.subplots(2, sharex=True)
    autocorrelation_plot(page_visits, ax=ax[0], color='b')
    ax[0].set_title("ACF - Page visits")
    autocorrelation_plot(first_order_diff.iloc[1:], ax=ax[1], color='r')
    ax[1].set_title("ACF - First-order differences to page visits")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
    plt.savefig('plots/2_acf.png')
    print("2_acf plot created\n")

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

def print_dicky_fuller_test_results(page_visits):
    for col in page_visits.columns:
        print("Column: {}".format(col))
        print("Null hypothesis is that it is not stationary.")
        time_series = pd.Series(page_visits[col])
        adf_result = stattools.adfuller(time_series, autolag='AIC')
        print('p-val of the ADF test in air miles flown:', adf_result[1])
        print(adf_result)


first_order_differencing_checks()
print_dicky_fuller_test_results(page_visits)
