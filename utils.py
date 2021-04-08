import numpy as np
import pandas as pd

def get_pct_change(val1, val2):
    return (val2-val1)/val1

def get_SR(monthly_returns, yearly_expected_return=0.0, row_duration=1):
    # row_duration = number of days between consecutive records
    # Generally there are 252 trading days in a year
    yearly_return_multiplier = (252)/row_duration
    yearly_expected_return_duration = yearly_expected_return/yearly_return_multiplier

    #monthly_returns = [(1+val)**yearly_return_multiplier-1 for val in monthly_returns]
    #monthly_returns = [val*yearly_return_multiplier for val in monthly_returns]
    numerator = np.mean(monthly_returns) - yearly_expected_return_duration
    sharpe_ratio = numerator/np.std(monthly_returns)
    sharpe_ratio = sharpe_ratio*np.sqrt(yearly_return_multiplier)

    return sharpe_ratio


def get_sortino(monthly_returns, yearly_expected_return=0.0, row_duration=1, mar=0.0):
    # mar: minimim acceptable return over full duration
    yearly_return_multiplier = (252)/row_duration
    mar_duration = mar/yearly_return_multiplier
    yearly_expected_return_duration = yearly_expected_return/yearly_return_multiplier
    #monthly_returns = [(1+val)**yearly_return_multiplier-1 for val in monthly_returns]
    #monthly_returns = np.array(monthly_returns)*yearly_return_multiplier

    numerator = (np.mean(monthly_returns)-yearly_expected_return_duration)
    duration = len(monthly_returns)
    monthly_returns = [val-mar_duration for val in monthly_returns]
    monthly_returns = [val*val for val in monthly_returns if val < 0]
    down_std = np.sqrt(np.sum(monthly_returns)/duration)
    sortino = numerator/down_std
    sortino = sortino*np.sqrt(yearly_return_multiplier)

    return sortino

def get_max_drawdown(account_vals):
    # given account values over a period calculates % max drawdown
    max_drawdown = 0 # is positive
    for i in range(len(account_vals)-1):
        dd = (account_vals[i] - min(account_vals[i+1:]))/account_vals[i]
        max_drawdown = max(max_drawdown, dd)

    return max_drawdown
