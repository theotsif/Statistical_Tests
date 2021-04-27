from matplotlib import pyplot
import numpy as np
from sklearn.utils import resample

def statistic_calc(statistic, arr):
    if statistic == 'median':
        return np.median(arr)
    elif statistic == 'mean':
        return np.mean(arr)
    else:
        raise ValueError('Chose either mean or median')


def bootstrap_diff(arr_a, arr_b, categories, statistic='median', n_iterations=10 ** 4, sample_size=1, alpha=0.05):
    '''
    Performs bootstrapping to estimate the confidence intervals for the difference of a chosen statistic (median or
    mean) between 2 groups. Sampling with replacement on both groups for a given ratio takes places and the chosen
    statistic is calculated; then the difference between the calculated statistics from both groups is taken.
    The process ic repeated for a chosen number of iterations (default value is 10,000).
    Finally for a given level of confidence the significance of the average difference is reported.
    '''

    group_a_count = len(arr_a)
    group_b_count = len(arr_b)

    # configure bootstrap
    n_size_a = int(group_a_count * sample_size)
    n_size_b = int(group_b_count * sample_size)

    diffs = list()

    for i in range(n_iterations):

        # sample groups
        group_a = resample(arr_a, n_samples=n_size_a)
        group_b = resample(arr_b, n_samples=n_size_b)

        # calculate statistics for the two groups
        statistic_a = statistic_calc(statistic, group_a)
        statistic_b = statistic_calc(statistic, group_b)

        # calculate the difference between the chosen statistics
        diff = statistic_a - statistic_b
        diffs.append(diff)

    A = sum([i > 0 for i in diffs])
    B = len(diffs)
    C = sum([i == 0 for i in diffs])
    p_star = A / B + 0.5 * (C / B)
    p_star_m = min(p_star, 1 - p_star)

    if p_star_m <= alpha / 2:

        print('Evaluation at ' + '%s' % (str(n_iterations) + ' Bootstrap Iterations with Sampling at '
                                         + str(100 * sample_size) + ' %' + ' of the original population: '
                                         + categories[0] + ' is greater than ' + categories[1] + ' and significant at {}'
                                         .format(str(round(1 - p_star_m, 3) * 100) + ' %')))
    else:
        print(categories[0] + ' is not greater than ' + categories[1] + ' at {}  significance'.format(
            str(round(1 - p_star_m, 3) * 100) + ' %'))

    return diffs, p_star_m
