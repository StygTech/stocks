import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.stats as stats
import scipy.linalg as sla
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as dates
import os

from stock_functions import values2returns, returns2values, get_clean_data, block_resample, ema, lse, predict


def study_leverage():
    # Block resampling w/ daily-reset leverage study
    leverage_ratio_list = [1.0, 2.0, 3.0, 4.0, 10.0]

    for leverage_ratio in leverage_ratio_list:
        xs = np.zeros([num_samples, nt+1])
        for i in range(num_samples):
            rs = leverage_ratio*block_resample(r, block_length, forward_limit, backward_limit)
            xs[i] = returns2values(rs)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(td, np.ones_like(x), color='k', alpha=0.5, linestyle='--', label='Baseline')
        plt.plot(td, x, color='k', alpha=0.9, label='Actual')
        if leverage_ratio != 1.0:
            x_lev = returns2values(leverage_ratio*r)
            plt.plot(td, x_lev, color='C4', alpha=0.9, label='Actual (Leveraged %.1fX)' % leverage_ratio)
        percentiles = [0, 1, 5, 25]
        num_levels = len(percentiles)
        cmap = cm.get_cmap('GnBu', 256)
        fill_colors = cmap(np.linspace(0.5, 0.9, num_levels))
        for percentile, fill_color in zip(percentiles, fill_colors):
            p_lwr = percentile
            p_upr = 100-percentile
            x_lwr = np.percentile(xs, p_lwr, axis=0)
            x_upr = np.percentile(xs, p_upr, axis=0)
            plt.fill_between(td, x_lwr, x_upr, color=fill_color, alpha=0.4, linewidth=0,
                             label='%3.0f-%3.0f Percentile' % (p_lwr, p_upr))
        plt.yscale('log')
        plt.xlabel('Date')
        plt.ylabel('Price')
        leg = plt.legend(loc='upper left')
        plt.setp(leg.texts, family='monospace')
        plt.title('Asset vs Time')
        plt.grid('on')
        plt.tight_layout()

    return


def study_returns_sequence():
    # Block resampling study of actual vs sorted vs flattened returns sequences

    # Sorted
    r_sort = np.sort(r)
    x_sort = returns2values(r_sort)

    # Flattened
    x_flat_tgt = np.logspace(np.log10(x[0]), np.log10(x[-1]), nt+1)
    r_flat_candidates = np.copy(r_sort)
    r_flat = np.zeros(nt)
    x_flat = np.ones(nt+1)
    for i in range(nt):
        x_flat_next_candidates = x_flat[i] * (1+(r_flat_candidates / 100))
        candidate_idx = np.argmin(np.abs(x_flat_next_candidates-x_flat_tgt[i+1]))
        r_flat[i] = r_flat_candidates[candidate_idx]
        x_flat[i+1] = x_flat[i] * (1+(r_flat[i] / 100))
        rsa_candidates = np.delete(r_flat_candidates, candidate_idx)

    # Show the actual value sequences together
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(td, x, color='k', alpha=0.9, label='Actual')
    plt.plot(td, x_sort, color='C0', alpha=0.9, label='Sorted')
    plt.plot(td, x_flat, color='C1', alpha=0.9, label='Flattened')

    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel('Price')
    leg = plt.legend(loc='upper left')
    plt.setp(leg.texts, family='monospace')
    plt.title('Asset vs Time')
    plt.grid('on')
    plt.tight_layout()

    # Show block bootstrap results
    for r_test, x_test in zip([r, r_sort, r_flat], [x, x_sort, x_flat]):
        xs = np.zeros([num_samples, nt+1])
        for i in range(num_samples):
            rs = block_resample(r_test, block_length, forward_limit, backward_limit)
            xs[i] = returns2values(rs)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(td, np.ones_like(x_test), color='k', alpha=0.5, linestyle='--', label='Baseline')
        plt.plot(td, x_test, color='k', alpha=0.9, label='Actual')
        percentiles = [0, 1, 5, 25]
        num_levels = len(percentiles)
        cmap = cm.get_cmap('GnBu', 256)
        fill_colors = cmap(np.linspace(0.5, 0.9, num_levels))
        for percentile, fill_color in zip(percentiles, fill_colors):
            p_lwr = percentile
            p_upr = 100-percentile
            x_lwr = np.percentile(xs, p_lwr, axis=0)
            x_upr = np.percentile(xs, p_upr, axis=0)
            plt.fill_between(td, x_lwr, x_upr, color=fill_color, alpha=0.4, linewidth=0,
                             label='%3.0f-%3.0f Percentile' % (p_lwr, p_upr))
        plt.yscale('log')
        plt.xlabel('Date')
        plt.ylabel('Price')
        leg = plt.legend(loc='upper left')
        plt.setp(leg.texts, family='monospace')
        plt.title('Asset vs Time')
        plt.grid('on')
        plt.tight_layout()

    return


def study_model_basic():
    # Modeling study (basic)

    # Analyze residuals
    r_predict = np.zeros_like(r)
    r_err = r-r_predict
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
    bin_mids = bins[0:-1]+bin_widths / 2
    ylims = [-1.0, 1.0]
    # linthreshy = 1e-2
    for i, (dist_name, dist_pretty_name, dist_color) in enumerate(zip(dist_names, dist_pretty_names, dist_colors)):
        dist, param = [dist_dict[dist_name][key] for key in ['dist', 'param']]
        pdf_fit_y = dist.pdf(bin_mids, *param[:-2], loc=param[-2], scale=param[-1])
        dist_err = counts-pdf_fit_y
        ax[i].bar(bin_mids, dist_err, label=dist_pretty_name, color=dist_color, lw=2, alpha=0.5, width=bin_widths)
        ax[i].set_ylim(ylims)
        # ax[i].set_yscale('symlog', linthreshy=linthreshy)
        ax[i].legend(loc='upper right')
    pdf_fit_y = kde.evaluate(bin_mids)
    dist_err = counts-pdf_fit_y
    i = -1
    ax[i].bar(bin_mids, dist_err, label='Gaussian KDE', color='C5', lw=2, alpha=0.5, width=bin_widths)
    ax[i].set_ylim(ylims)
    # ax[i].set_yscale('symlog', linthreshy=linthreshy)
    ax[i].set_xlabel('Return (%)')
    ax[i].legend(loc='upper right')
    ax[0].set_title('Probability Density Error')
    fig.tight_layout()


    # Generate new trajectories using the fitted noise model
    dist_name = 'gennorm'
    dist = dist_dict[dist_name]['dist']
    param = dist_dict[dist_name]['param']
    vs = np.zeros([num_samples, nt+1])
    for i in range(num_samples):
        rs = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=nt)
        vs[i] = returns2values(rs)

    percentiles = [0, 1, 5, 25]
    num_levels = len(percentiles)
    cmap = cm.get_cmap('GnBu', 256)
    fill_colors = cmap(np.linspace(0.5, 0.9, num_levels))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(td, x, color='k', alpha=0.9, label='Actual')
    # plt.plot(td, x_predict, color='C4', alpha=0.9, label='Predicted')
    for percentile, fill_color in zip(percentiles, fill_colors):
        p_lwr = percentile
        p_upr = 100-percentile
        x_lwr = np.percentile(vs, p_lwr, axis=0)
        x_upr = np.percentile(vs, p_upr, axis=0)
        plt.fill_between(td, x_lwr, x_upr, color=fill_color, alpha=0.4, linewidth=0,
                         label='%3.0f-%3.0f Percentile' % (p_lwr, p_upr))
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel('Price')
    leg = plt.legend(loc='upper left')
    plt.setp(leg.texts, family='monospace')
    plt.title('Asset vs Time')
    plt.grid('on')
    plt.tight_layout()

    return


def study_model_ar():
    # Modeling study (Autoregressive model)
    # May be slow if using full 90 years of data, try with just 10 years first

    # Fit an autoregressive model
    ar_length = 30
    predict_params = lse(r, n=ar_length)
    r_predict = predict(r, predict_params)
    x_predict = returns2values(r_predict)

    fig, ax = plt.subplots(nrows=2, figsize=(8, 10))
    ax[0].plot(td[1:], r, label='Actual')
    ax[0].plot(td[1:], r_predict, label='Predicted')
    ax[0].legend()
    ax[0].set_title('Raw')

    ema_alpha = 0.01
    ax[1].plot(td[1:], ema(r, ema_alpha, y0=0), label='Actual')
    ax[1].plot(td[1:], ema(r_predict, ema_alpha, y0=0), label='Predicted')
    ax[1].legend()
    ax[1].set_title(r'Exponential moving averages, $\alpha = %0.3f$' % ema_alpha)


    # Analyze residuals after removing predicted values
    r_err = r - r_predict
    rmse = np.sqrt(np.mean(r_err**2))
    print('root-mean-square error: %f' % rmse)

    fig, ax = plt.subplots(figsize=(8, 6))
    r_upr = np.percentile(np.abs(r_err), 99.9)
    r_lwr = -r_upr
    r_range = (r_lwr, r_upr)
    counts, _, _ = plt.hist(r_err, bins=100, range=r_range, density=True, color='C0', alpha=0.5, label='Empirical')
    dist_names = ['norm', 'gennorm', 'cauchy']
    dist_pretty_names = ['Normal', 'Generalized normal', 'Cauchy']
    dist_colors = ['C1', 'C0', 'C3']
    dist_dict = {}
    for dist_name, dist_pretty_name, dist_color in zip(dist_names, dist_pretty_names, dist_colors):
        dist = getattr(stats, dist_name)
        param = dist.fit(r_err)
        dist_dict[dist_name] = {'dist': dist,
                                'param': param}
        pdf_fit_x = np.linspace(r_range[0], r_range[1], 1000)
        pdf_fit_y = dist.pdf(pdf_fit_x, *param[:-2], loc=param[-2], scale=param[-1])
        plt.plot(pdf_fit_x, pdf_fit_y, label=dist_pretty_name, color=dist_color)
    plt.ylim([0.5*np.min(counts[counts > 0]), 2.0*np.max(counts[counts > 0])])
    plt.yscale('log')
    plt.legend()


    #
    # # Model-based method 1 (cheating?)
    # # Generate new trajectories using the fitted prediction and noise models
    # dist_name = 'gennorm'
    # dist = dist_dict[dist_name]['dist']
    # param = dist_dict[dist_name]['param']
    # vs = np.zeros([num_samples, nt+1])
    # for i in range(num_samples):
    #     r_err_s = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=nt)
    #     rs = r_predict + r_err_s
    #     vs[i] = returns2values(rs)
    #
    #

    # Model-based method 2
    def predict2(r_err, a):
        T = r_err.size
        y = np.zeros(T)
        n = a.size-1
        for i in range(T-1):
            if i >= n:
                ym = np.hstack([y[i:i-n:-1], 1.0])
            else:
                ym = np.hstack([y[i::-1], np.zeros(n-i-1), 1.0])
            y[i+1] = np.dot(a, ym) + r_err[i]
        return y

    # Generate new trajectories using the fitted prediction and noise models
    dist_name = 'gennorm'
    dist = dist_dict[dist_name]['dist']
    param = dist_dict[dist_name]['param']

    vs = np.zeros([num_samples, nt+1])
    for i in range(num_samples):
        r_err_s = dist.rvs(*param[:-2], loc=param[-2], scale=param[-1], size=nt)
        rs = predict2(r_err_s, predict_params)
        vs[i] = returns2values(rs)

    percentiles = [0, 1, 5, 25]
    num_levels = len(percentiles)
    cmap = cm.get_cmap('GnBu', 256)
    fill_colors = cmap(np.linspace(0.5, 0.9, num_levels))
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(td, x, color='k', alpha=0.9, label='Actual')
    # plt.plot(td, x_predict, color='C4', alpha=0.9, label='Predicted')
    for percentile, fill_color in zip(percentiles, fill_colors):
        p_lwr = percentile
        p_upr = 100-percentile
        x_lwr = np.percentile(vs, p_lwr, axis=0)
        x_upr = np.percentile(vs, p_upr, axis=0)
        plt.fill_between(td, x_lwr, x_upr, color=fill_color, alpha=0.4, linewidth=0,
                         label='%3.0f-%3.0f Percentile' % (p_lwr, p_upr))
    plt.yscale('log')
    plt.xlabel('Date')
    plt.ylabel('Price')
    leg = plt.legend(loc='upper left')
    plt.setp(leg.texts, family='monospace')
    plt.title('Asset vs Time')
    plt.grid('on')
    plt.tight_layout()

    return


def study_bandit():
    # Multi-armed bandit study

    data = {}
    ticker_list = ['VFINX', 'VWESX']
    for ticker in ticker_list:
        t_hist, x_hist = get_clean_data(ticker, dir, fmt, lwr, upr)
        r_hist = values2returns(x_hist)
        data[ticker] = {'t': t_hist,
                        'x': x_hist,
                        'r': r_hist}

    nt = t_hist.size - 1
    td_hist = np.array([dates.num2date(t_hist[i]) for i in range(nt+1)])
    # t0_hist = t_hist - t_hist[0]

    def reward(a, t):
        ticker = ticker_list[a]
        return data[ticker]['r'][t]


    # Number of arms of the bandit
    k = len(ticker_list)

    # True means
    q_opt = np.array([np.mean(data[ticker]['r']) for ticker in ticker_list])

    # Optimal action
    a_opt = np.argmax(q_opt)


    # Epsilon-greedy policy
    def epsilon_greedy(q, t, n, epsilon=0.1):
        a = np.argmax(q)
        if npr.rand() < epsilon:
            i = npr.randint(k-1)
            a = np.arange(k)[i if i < a else i+1]
        return a


    # Upper confidence bound policy
    def upper_confidence_bound(q, t, n, c=1.0):
        a = np.argmax(q+c * (2 * np.log(t) / (n+1))**0.5)
        return a


    def experiment(policy, parameter):
        # Initialize
        q = np.zeros(k)
        n = np.zeros(k)

        # History
        a_hist = np.zeros(nt)
        r_hist = np.zeros(nt)
        q_hist = np.zeros([nt, k])
        n_hist = np.zeros([nt, k])
        t_hist = np.arange(nt)

        # Iterate
        for t in t_hist:
            # Generate action
            a = policy(q, t+1, n, parameter)

            # Generate reward
            r = reward(a, t)

            # Accumulate number of plays of chosen action
            n[a] += 1

            # Record history
            a_hist[t] = a
            r_hist[t] = r
            q_hist[t] = q
            n_hist[t] = n

            # Update action-value function using incremental sample-average rule
            # q[a] += (r - q[a]) / n[a]

            # Update action-value function using knowledge of reward of all actions
            for a in range(k):
                q[a] += (reward(a, t) - q[a]) / (t+1)

        return q, a_hist, r_hist, q_hist, n_hist, t_hist

    policy = epsilon_greedy
    epsilon = 0.1

    num_experiments = 50
    a_hist = np.zeros([num_experiments, nt])
    r_hist = np.zeros([num_experiments, nt])
    q_hist = np.zeros([num_experiments, nt, k])
    n_hist = np.zeros([num_experiments, nt, k])
    t_hist = np.arange(nt)
    for i in range(num_experiments):
        q, a_hist[i], r_hist[i], q_hist[i], n_hist[i], t_hist = experiment(policy, epsilon)


    # Plotting
    plt.figure()
    # plt.plot(td_hist[1:], np.cumsum(r_hist), label=r'$\epsilon$-greedy, %.3e' % epsilon)
    plt.plot(td_hist[1:], np.cumsum(r_hist, axis=1).T, color='k', alpha=0.1)
    for ticker in ticker_list:
        plt.plot(td_hist[1:], np.cumsum(data[ticker]['r']), label=ticker)
    leg = plt.legend(loc='upper right')
    plt.setp(leg.texts, family='monospace')
    plt.grid('on')


    experiment_data_list = []
    policy_list = [epsilon_greedy, epsilon_greedy, epsilon_greedy, upper_confidence_bound]
    parameter_list = [0.002, 0.02, 0.2, 1.00]
    policy_strings = ['$\epsilon$-greedy, $\epsilon=0.002$',
                      '$\epsilon$-greedy, $\epsilon=0.020$',
                      '$\epsilon$-greedy, $\epsilon=0.200$',
                      'upper confidence bound']

    for policy, parameter in zip(policy_list, parameter_list):
        experiment_data_list.append(experiment(policy, parameter))

    plt.figure()
    x = np.array([data[ticker]['r'] for ticker in ticker_list]).T
    bp = plt.boxplot(x)
    plt.xlabel('Arm')
    plt.ylabel('Reward')
    plt.title('Bandit rewards')
    plt.show()


    num_plots = 3
    fig_list, ax_list = [], []
    for i in range(num_plots):
        fig, ax = plt.subplots()
        fig_list.append(fig)
        ax_list.append(ax)

    for i, policy_string in enumerate(policy_strings):
        q, a_hist, r_hist, q_hist, n_hist, t_hist = experiment_data_list[i]

        # Reward history
        ax_list[0].plot(t_hist, np.cumsum(r_hist) / (t_hist+1))

        # Expected regret history
        r_hist_expected = np.array([q_opt[a] for a in a_hist.astype(int)])
        regret = q_opt.max()-r_hist_expected
        ax_list[1].plot(t_hist, np.cumsum(regret))

        # Action-value function estimation error
        ax_list[2].semilogy(t_hist, la.norm(q_hist-q_opt, axis=1))

    ax_list[0].plot(t_hist, np.max(q_opt) * np.ones(nt), 'k--')

    title_str_list = ['Time-averaged total reward vs time',
                      'Total expected regret vs time',
                      'Action-value estimation error vs time']
    for ax, title_str in zip(ax_list, title_str_list):
        ax.set_title(title_str)
        ax.legend(policy_strings)
    plt.show()

    return


if __name__ == "__main__":
    seed = 1
    npr.seed(1)
    plt.close('all')
    plt.style.use('./fivethirtyeight_mod.mplstyle')

    # Import and pre-process data
    dir = 'yahoo'
    # dir = 'quantquote_daily_sp500'

    fmt = 'yahoo'
    # fmt = 'quantquote'

    ticker = '^GSPC'  # S&P 500 index
    # ticker = 'VFINX'  # Vanguard 500 Index Fund
    # ticker = 'VWESX'  # Vanguard Long-Term Investment-Grade Bond Fund
    # ticker = 'LMT'  # Lockheed Martin

    lwr, upr = None, None
    # lwr, upr = -40*365, None  # Last 40 years
    # lwr, upr = -10*365, None  # Last 10 years

    t, x = get_clean_data(ticker, dir, fmt, lwr, upr)

    nt = t.size - 1
    td = np.array([dates.num2date(t[i]) for i in range(nt+1)])
    r = values2returns(x)

    # Monte Carlo simulation settings
    num_samples = 1000

    # block_length = 1*365
    # forward_limit = None
    # backward_limit = None

    # block_length = 4*365
    # forward_limit = 8*365
    # backward_limit = 8*365

    block_length = 2*365
    forward_limit = 4*365
    backward_limit = 4*365

    # block_length = 1*365
    # forward_limit = 1*365
    # backward_limit = 1*365

    # Run studies
    study_leverage()
    # study_returns_sequence()
    # study_model_basic()
    # study_model_ar()
    # study_bandit()

