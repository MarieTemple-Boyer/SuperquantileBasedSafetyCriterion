""" Compute empircal probability of rejecting H_0. """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from superquantile import superquantile

FILENAME = 'borehole.pkl'

SAMPLE_SIZE = 5*10**4
ALPHA = 2*10**(-2)
BETA = 10**(-2)

N_ESTIM_P = 2*10**3

OUTPUT_MIN = 9.91
OUTPUT_MAX = 309.58
OUTPUT_RANGE = OUTPUT_MAX - OUTPUT_MIN

N_VALS = 10**4

''' Graphical parameters '''
font = {'size': 15}
# font = {'size': 15}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

ORD_MAX = 1.15
ORD_MIN = -0.02
ABS_MIN = OUTPUT_MIN
ABS_MAX = OUTPUT_MAX

ORD_ARROW = 0.7

with open(FILENAME, 'rb') as f:
    sample_huge = pickle.load(f)

sample = sample_huge[:SAMPLE_SIZE]


eta_bar = OUTPUT_RANGE * \
    np.sqrt((5 * np.log(3/BETA)) / (ALPHA*SAMPLE_SIZE))

estim_sup = superquantile(sample, alpha=1-ALPHA)

real_sup = superquantile(sample_huge, alpha=1-ALPHA)

def upper_brown(threshold, sup, eta, alpha, sample_size):
    ''' Return an upper bound of the probability of labeling as safe. '''
    gap = threshold-sup-eta
    res = 3*np.exp(-alpha * gap**2 * sample_size /
                   (5*OUTPUT_RANGE**2)) * (gap < 0) + (gap >= 0)
    return 1-(1-res)*(res <= 1)


def lower_brown(threshold, sup, eta, alpha, sample_size):
    ''' Return a lower bound of the probability of labeling as safe. '''
    gap = threshold-sup-eta
    return (1-np.exp(-2 * alpha**2 * gap**2 * sample_size / OUTPUT_RANGE**2)) \
        * (gap >= 0)


def lower_wang(threshold, sup, eta, alpha, sample_size):
    ''' Return a lower bound of the probability of labeling as safe. '''
    gap = threshold-sup-eta
    res = (1-3*np.exp(-1/11 * alpha * gap**2 * sample_size \
                            / OUTPUT_RANGE**2)) \
                * (gap >= 0)
    return res * (res >= 0)



thresholds = np.linspace(max(OUTPUT_MIN, ABS_MIN),
                         min(OUTPUT_MAX, ABS_MAX),
                         N_VALS)

prob_max = upper_brown(thresholds, real_sup, eta_bar, ALPHA, SAMPLE_SIZE)

low_brown = lower_brown(thresholds, real_sup, eta_bar, ALPHA, SAMPLE_SIZE)
low_wang = lower_wang(thresholds, real_sup, eta_bar, ALPHA, SAMPLE_SIZE)

prob_min = np.zeros((2, N_VALS))
prob_min[0, :] = low_brown
prob_min[1, :] = low_wang
prob_min = np.max(prob_min, axis=0)


# Computation of real distribution and rejection probability
assert len(sample_huge)>=SAMPLE_SIZE*N_ESTIM_P

tab_estim_sup = np.zeros(N_ESTIM_P)
for i in range(N_ESTIM_P):
    tab_estim_sup[i] = superquantile(
                        sample_huge[i*SAMPLE_SIZE:(i+1)*SAMPLE_SIZE],
                        alpha=1-ALPHA)
real_prob_rej = np.sum(tab_estim_sup[:, np.newaxis]+eta_bar < thresholds,
                        axis=0)/N_ESTIM_P


# Approximation using bootstrap
tab_estim_sup_bootstrap = np.zeros(N_ESTIM_P)
for i in range(N_ESTIM_P):
    sample_bootstrap = np.random.choice(sample, size=SAMPLE_SIZE, replace=True)
    tab_estim_sup_bootstrap[i] = superquantile(sample_bootstrap, alpha=1-ALPHA)
estim_prob_rej = np.sum(tab_estim_sup_bootstrap[:, np.newaxis]+eta_bar\
                                < thresholds,
                        axis=0)/N_ESTIM_P

fig, ax = plt.subplots(figsize=(5.6, 3.2))

fig.subplots_adjust(left=0.12, right=0.89, top=1, bottom=0.2)

ax.set_xlim(left=145, right=300)

ax_bis = ax.twinx()


ax.fill_between(thresholds,
                prob_min,
                prob_max,
                color='gray',
                alpha=0.2)

# Real values
ax.plot(thresholds,
        real_prob_rej,
        color='black', linestyle='dotted',
        label=r'$\mathbb{P} \left ( \overline{W}_n \right )$')

counts, bins, _ = ax_bis.hist(tab_estim_sup, density=True,
                              color='orange', alpha=0.9)
#ax_bis.hist(tab_estim_sup_bootstrap, density=True, color='orange', alpha=0.5)



top_ax_bis = np.max(counts)*2

ax.set_ylim(bottom=ORD_MIN, top=ORD_MAX)
ax_bis.set_ylim(bottom = top_ax_bis*ORD_MIN/ORD_MAX, top=top_ax_bis)

# Superquantile
ax.plot([estim_sup, estim_sup],
        [0,1],
        color='blue')
ax.annotate(r'$\widehat{\overline{q}}_{1-\alpha, n}$',
            [estim_sup, 1],
            ha='left', va='bottom', color='blue')

ax.plot([real_sup, real_sup],
        [0,1],
        color='black', linestyle='dashed')
ax.annotate(r'$\overline{q}_{1-\alpha}$',
            [real_sup, 1],
            ha='right', va='bottom')

# arrow
ax.annotate('', xy=(real_sup, ORD_ARROW),
            xytext=(real_sup+eta_bar, ORD_ARROW),
            arrowprops={'arrowstyle': '<->', 'color': 'black'})
ax.annotate(r'$\overline{\eta}$', xy=(real_sup+eta_bar/2, ORD_ARROW),
            color='black', ha='center', va='bottom')


# Probability of rejecting
ax.plot(thresholds,
        estim_prob_rej,
        color='red',
        label=r'$\mathbb{P} \left ( \overline{W}_n \right )$ (bootstrap)')

ax.grid()


ax.set_xlabel(r'Threshold ($s$)')
ax.set_ylabel(r'Probability of rejecting $H_0$')
ax_bis.set_ylabel('Scale for histogram')
ax.legend(loc='lower right')

plt.show()
