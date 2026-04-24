""" Plot histogram. """

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from superquantile import superquantile

FILENAME = 'borehole.pkl'

SAMPLE_SIZE = 10**2
ALPHA = 2*10**(-2)

BETA = 10**(-2)

THRESHOLD = 230

OUTPUT_MIN = 9.91
OUTPUT_MAX = 309.58
OUTPUT_RANGE = OUTPUT_MAX - OUTPUT_MIN

''' Graphical parameters '''
font = {'size': 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
ORD_ARROW = 0.01


with open(FILENAME, 'rb') as f:
    sample_huge = pickle.load(f)

sample = sample_huge[:SAMPLE_SIZE]

eta_bar = OUTPUT_RANGE * np.sqrt((5 * np.log(3/BETA)) / (ALPHA*SAMPLE_SIZE))

estim_sup = superquantile(sample, alpha=1-ALPHA)

ncrit = 5/ALPHA*np.log(3/BETA) * (OUTPUT_RANGE/(THRESHOLD-OUTPUT_MIN))**2


print(f'critical sample size: {ncrit}')
print(f'eta_bar: {eta_bar}')
print(f's-eta_bar: {THRESHOLD-eta_bar}')
print(f'estim sup: {estim_sup}')

fig, ax = plt.subplots(figsize=(5.6, 3.2))

plt.subplots_adjust(left=0.2, right=1, top=1, bottom=0.2)


NBINS = 100

ax.hist(sample, bins=NBINS, density=True, color='grey')
_, xlim_right = ax.get_xlim()
_, ylim_left = ax.set_ylim()
ax.plot([THRESHOLD, THRESHOLD], [0, ylim_left], color='blue')
ax.annotate(r's', [THRESHOLD, 0], color='blue', va='top', ha='center')

ax.plot([THRESHOLD-eta_bar, THRESHOLD-eta_bar], [0, ylim_left],
        color='blue', linestyle='dashed')
# ax.annotate(r'$s - \overline{\eta}$', [THRESHOLD-eta_bar, 0],
#            va='top', ha='center')

ax.plot([estim_sup, estim_sup], [0, ylim_left], color='black')
ax.annotate(r'$\widehat{\overline{q}}_{1-\alpha, n}$', [estim_sup, 0],
            va='top', ha='center')

ax.annotate('', xy=(THRESHOLD, ORD_ARROW),
            xytext=(THRESHOLD-eta_bar, ORD_ARROW),
            arrowprops={'arrowstyle': '<->', 'color': 'red'})
ax.annotate(r'$\overline{\eta}$', xy=(THRESHOLD-eta_bar/2, ORD_ARROW),
            color='red', ha='center', va='bottom')

ax.set_xlabel('Y')
ax.set_ylabel('Density')

ax.grid()
plt.show()
