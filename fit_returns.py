import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.stats as stats
import scipy.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as dates
import os


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


def import_raw_data(path, fmt=None):
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

    # Get price time-series
    x_open_raw = np.array([data[i][open_column] for i in range(1, nt_raw+1)])
    x_high_raw = np.array([data[i][high_column] for i in range(1, nt_raw+1)])
    x_low_raw = np.array([data[i][low_column] for i in range(1, nt_raw+1)])
    x_close_raw = np.array([data[i][close_column] for i in range(1, nt_raw+1)])

    # Compute daily prices as the average of open and close
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
    t_raw, x_raw = import_raw_data(path_in, fmt)
    t, x = clean_data(t_raw, x_raw, lwr, upr)
    return t, x


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = stats.gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


if __name__ == "__main__":
    # Import and pre-process data
    dir = 'yahoo'
    fmt = 'yahoo'

    # Choose a ticker's historical data to import
    ticker = '^GSPC'  # S&P 500 index
    # ticker = 'VFINX'  # Vanguard 500 Index Fund
    # ticker = 'VWESX'  # Vanguard Long-Term Investment-Grade Bond Fund
    # ticker = 'LMT'  # Lockheed Martin

    # Set the date range
    lwr, upr = None, None
    # lwr, upr = -40*365, None  # Last 40 years
    # lwr, upr = -10*365, None  # Last 10 years

    t, x = get_clean_data(ticker, dir, fmt, lwr, upr)

    nt = t.size - 1
    td = np.array([dates.num2date(t[i]) for i in range(nt+1)])
    r = values2returns(x)

    # Analyze residuals
    r_predict = np.zeros_like(r)
    r_err = r_predict - r
    r_upr = np.percentile(np.abs(r_err), 100-0.1)
    r_lwr = -r_upr
    r_range = (r_lwr, r_upr)

    dist_names = ['norm', 'laplace', 'gennorm', 'cauchy']
    dist_pretty_names = ['Normal', 'Laplace', 'Generalized normal', 'Cauchy']
    dist_colors = ['C1', 'C2', 'C0', 'C3']
    dist_dict = {}
    num_dists = len(dist_names)

    # Fit parametric distributions
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(r_err)
        dist_dict[dist_name] = {'dist': dist, 'param': param}

    # Fit Gaussian kernel density estimator
    bandwidth = 0.2
    kde = stats.gaussian_kde(r_err, bw_method=bandwidth / r_err.std(ddof=1))


    # Plotting
    plt.close('all')
    plt.style.use('./fivethirtyeight_mod.mplstyle')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the empirical distribution
    counts, bins, patches = plt.hist(r_err, bins=100, range=r_range, density=True,
                                            color='C0', alpha=0.5, label='Empirical')

    # Plot fitted parametric distributions
    pdf_fit_x = np.linspace(r_range[0], r_range[1], 1000)
    for dist_name, dist_pretty_name, dist_color in zip(dist_names, dist_pretty_names, dist_colors):
        dist, param = [dist_dict[dist_name][key] for key in ['dist', 'param']]
        pdf_fit_y = dist.pdf(pdf_fit_x, *param[:-2], loc=param[-2], scale=param[-1])
        plt.plot(pdf_fit_x, pdf_fit_y, label=dist_pretty_name, color=dist_color, lw=2)

    # Plot Gaussian kernel density estimate distribution
    pdf_fit_y = kde.evaluate(pdf_fit_x)
    plt.plot(pdf_fit_x, pdf_fit_y, label='Gaussian KDE', color='C5', lw=2)

    plt.ylim([0.5 * np.min(counts[counts > 0]), 2.0 * np.max(counts[counts > 0])])
    plt.yscale('log')
    plt.xlabel('Return (%)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()

    # Plot the error in distribution
    fig, ax = plt.subplots(nrows=num_dists+1, sharex=True, sharey=True, figsize=(8, 8))
    bin_widths = bins[1:]-bins[0:-1]
    bin_mids = bins[0:-1] + bin_widths/2
    ylims = [-1.0, 1.0]
    # linthreshy = 1e-2
    for i, (dist_name, dist_pretty_name, dist_color) in enumerate(zip(dist_names, dist_pretty_names, dist_colors)):
        dist, param = [dist_dict[dist_name][key] for key in ['dist', 'param']]
        pdf_fit_y = dist.pdf(bin_mids, *param[:-2], loc=param[-2], scale=param[-1])
        dist_err = counts - pdf_fit_y
        ax[i].bar(bin_mids, dist_err, label=dist_pretty_name, color=dist_color, lw=2, alpha=0.5, width=bin_widths)
        ax[i].set_ylim(ylims)
        # ax[i].set_yscale('symlog', linthreshy=linthreshy)
        ax[i].legend(loc='upper right')
    pdf_fit_y = kde.evaluate(bin_mids)
    dist_err = counts - pdf_fit_y
    i = -1
    ax[i].bar(bin_mids, dist_err, label='Gaussian KDE', color='C5', lw=2, alpha=0.5, width=bin_widths)
    ax[i].set_ylim(ylims)
    # ax[i].set_yscale('symlog', linthreshy=linthreshy)
    ax[i].set_xlabel('Return (%)')
    ax[i].legend(loc='upper right')
    ax[0].set_title('Probability Density Error')
    fig.tight_layout()
