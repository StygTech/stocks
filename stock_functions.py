import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.stats as stats
import scipy.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as dates
import os


def import_data(path, fmt=None):
    # TODO support quantquote format - fix columns, date string format
    if fmt is None:
        fmt = 'yahoo'
    if fmt == 'yahoo':
        data_dtype = "S10,f8,f8,f8,f8,f8,f8"
        date_column, open_column, high_column, low_column, close_column = 0, 1, 2, 3, 4
    elif fmt == 'quantquote':
        data_dtype = "S10,f8,f8,f8,f8,f8,f8,f8"
        date_column, open_column, high_column, low_column, close_column = 0, 2, 3, 4, 5
    else:
        raise ValueError

    # Extract the raw data and prepare it
    data = np.genfromtxt(path, dtype=data_dtype, delimiter=',')
    nt_raw = len(data)-1

    dates_raw = np.array([data[i][date_column] for i in range(1, nt_raw+1)])
    t_raw = np.array([dates.datestr2num(''.join(dates_raw[i].decode('UTF-8').split('-'))) for i in range(nt_raw)])

    x_open_raw = np.array([data[i][open_column] for i in range(1, nt_raw+1)])
    x_high_raw = np.array([data[i][high_column] for i in range(1, nt_raw+1)])
    x_low_raw = np.array([data[i][low_column] for i in range(1, nt_raw+1)])
    x_close_raw = np.array([data[i][close_column] for i in range(1, nt_raw+1)])
    x_raw = (x_open_raw+x_close_raw) / 2

    return t_raw, x_raw


def clean_data(t_raw, x_raw, lwr=None, upr=None):
    t, x = fill_missing(t_raw, x_raw)
    t, x = t[lwr:upr], x[lwr:upr]
    x = x/x[0]
    return t, x


def get_clean_data(ticker, dir=None, fmt=None, lwr=None, upr=None):
    if dir is None:
        dir = '.'
    if fmt == 'quantquote':
        ticker = 'table_' + ticker.lower()
    filename = ticker + '.csv'
    path_in = os.path.join(dir, filename)
    t_raw, x_raw = import_data(path_in, fmt)
    t, x = clean_data(t_raw, x_raw, lwr, upr)
    return t, x


def values2returns(v):
    x = v / v[0]
    dx = x[1:] / x[:-1]
    r = 100*(dx-1.0)
    return r


def returns2values(r):
    return np.cumprod(1.0 + np.hstack([0.0, r])/100)


def fill_missing(t, x):
    # Fill in missing days
    T = int(t[-1] - t[0])
    tn = int(t[0]) + np.arange(T)
    xn = np.interp(tn, t, x).round(2)
    return tn, xn


def block_resample(data, block_length, forward_limit=None, backward_limit=None, seed=None):
    # Set the random number generator
    rng = npr.RandomState(seed)
    # Get sizes and initialize resampled_data array
    data_length = len(data)
    num_blocks = np.ceil(data_length/block_length).astype(int)
    resampled_data = np.zeros(data_length)
    for i in range(num_blocks):
        # Compute the index limits in resampled_data where we will place the sampled block
        idx_lwr = i*block_length
        idx_upr = min((i+1)*block_length, data_length)
        # Compute the block length, truncated if we run past the end of the resampled_data length
        block_length_i = min(block_length, idx_upr-idx_lwr)
        # Determine the offset sample limits
        if backward_limit is not None:
            offset_lwr = max(idx_lwr-backward_limit, 0)
        else:
            offset_lwr = 0
        if forward_limit is not None:
            offset_upr = min(idx_upr+forward_limit, data_length) - block_length_i
        else:
            offset_upr = data_length-block_length_i
        # Randomly sample an offset index from the appropriate range
        if offset_upr > offset_lwr:
            offset = rng.randint(offset_lwr, offset_upr)
        else:
            offset = offset_lwr
        # Extract the block from data based on the sampled offset and put into the resampled_data array
        resampled_data[idx_lwr:idx_upr] = data[offset:offset+block_length_i]
    return resampled_data


def ema(x, a, y0=None):
    n = len(x)
    y = np.zeros(n)
    if y0 is None:
        y[0] = x[0]
    else:
        y[0] = y0
    for i in range(n-1):
        y[i+1] = a*x[i+1] + (1-a)*y[i]
    return y


def lse(x, n, rcond=None):
    T = x.size-1
    D = np.hstack([sla.toeplitz(x[n-1:T], x[n-1::-1]), np.ones([T-n+1, 1])])
    X = x[n:]
    return la.lstsq(D, X, rcond=rcond)[0]


def predict(x, a):
    y = np.zeros_like(x)
    T = x.size-1
    n = a.size-1
    for i in range(T-1):
        if i >= n:
            xm = np.hstack([x[i:i-n:-1], 1.0])
        else:
            xm = np.hstack([x[i::-1], np.zeros(n-i-1), 1.0])
        y[i+1] = np.dot(a, xm)
    return y


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = stats.gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)
