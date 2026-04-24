""" Upper and lower bounds for probability of rejecting H_0. """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

ALPHA = 0.02
BETA = 0.01

THRESHOLD = 230

SAMPLE_SIZES_EXP = [3, 4, 5, 7]
SAMPLE_SIZES_FAC = [5, 5, 5, 1]
SAMPLE_SIZES = [SAMPLE_SIZES_FAC[i]*10**SAMPLE_SIZES_EXP[i]
                for i in range(len(SAMPLE_SIZES_EXP))]

MIN = 9.91
MAX = 309.58

''' Graphical parameters '''
font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def inverse_power10(n):
    """ If n=10**a then return a. """
    return int(np.round(np.log(n)/np.log(10)))


def eta_bar(alpha, beta, sample_size, output_range):
    ''' Compute an upper bound of eta_* '''
    return output_range \
        * np.sqrt((5 * np.log(3/beta)) / (alpha*sample_size))


def upper_brown(superquantile, eta, alpha, sample_size, output_range):
    ''' Return Brown's upper bound of the probability of labeling as safe. '''
    gap = THRESHOLD-superquantile-eta
    res = 3*np.exp(-alpha * gap**2 * sample_size /
                   (5*output_range**2)) * (gap < 0) + (gap >= 0)
    return 1-(1-res)*(res <= 1)


def lower_brown(superquantile, eta, alpha, sample_size, output_range):
    ''' Return Brown's lower bound of the probability of labeling as safe. '''
    gap = THRESHOLD-superquantile-eta
    return (1-np.exp(-2 * alpha**2 * gap**2 * sample_size / output_range**2)) \
                * (gap >= 0)


def lower_wang(superquantile, eta, alpha, sample_size, output_range):
    ''' Return Wang's lower bound of the probability of labeling as safe. '''
    gap = THRESHOLD-superquantile-eta
    res = (1-3*np.exp(-1/11 * alpha * gap**2 \
            * sample_size / output_range**2)
           ) * (gap >= 0)
    return res * (res >= 0)


cmap = plt.get_cmap('tab10')
linestyles = ['dotted', 'dashed', 'solid', 'dashdot']

N = 10**4

sup_val = np.linspace(MIN, MAX, N)

fig, ax = plt.subplots(figsize=(7.5, 3.7))

fig.subplots_adjust(left=0.11, right=1, top=1, bottom=0.18)

for k, s_size in enumerate(SAMPLE_SIZES):
    eta_bar_const = eta_bar(ALPHA, BETA, s_size, MAX-MIN)
    low_brown = lower_brown(sup_val, eta_bar_const,
                            ALPHA, s_size, MAX-MIN)
    low_wang = lower_wang(sup_val, eta_bar_const, ALPHA, s_size, MAX-MIN)
    prob_min = np.zeros((2, N))
    prob_min[0, :] = low_brown
    prob_min[1, :] = low_wang
    prob_min = np.max(prob_min, axis=0)
    prob_max = upper_brown(sup_val, eta_bar_const, ALPHA, s_size, MAX-MIN)
    ax.plot(sup_val,
            prob_min,
            label=fr'${SAMPLE_SIZES_FAC[k]}\times10^{SAMPLE_SIZES_EXP[k]}$',
            color=cmap(k),
            linestyle=linestyles[k])
    ax.plot(sup_val,
            prob_max,
            color=cmap(k),
            linestyle=linestyles[k])
    ax.fill_between(sup_val,
                    prob_min,
                    prob_max,
                    color=cmap(k),
                    alpha=0.2)
ax.plot([THRESHOLD, THRESHOLD], [0, 1], color='black')
ax.plot([THRESHOLD, MAX], [BETA, BETA], color='black')
ax.plot([MIN, THRESHOLD], [BETA, BETA], color='gray')

ax.annotate(r'$s$', xy=(THRESHOLD, 0), ha='center', va='top')
ax.annotate(r'$\beta$', xy=(MIN, BETA), ha='right', va='bottom', color='gray')

ax.set_ylabel(r'Probability of rejecting $H_0$')
ax.set_xlabel(r'$\overline{q}_{1-\alpha}$')
ax.set_ylim(bottom=-0.15, top=1.15)
ax.set_xlim(left=MIN, right=MAX)

ax.grid()

fig.legend(title='Sample size')


plt.show()
