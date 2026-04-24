""" Generates outputs. """

import pickle
import openturns as ot
import numpy as np

SAMPLE_SIZE = 10**8

FILENAME = 'borehole.pkl'

r_omega = ot.TruncatedDistribution(
    ot.TruncatedDistribution(
        ot.Normal(0.1, 0.0161812),
        0.05,
        ot.TruncatedDistribution.LOWER),
    0.15,
    ot.TruncatedDistribution.UPPER)

r = ot.TruncatedDistribution(
    ot.TruncatedDistribution(
        ot.LogNormal(7.71, 0.0161812),
        100,
        ot.TruncatedDistribution.LOWER),
    50000,
    ot.TruncatedDistribution.UPPER)

T_u = ot.Uniform(63070, 115600)
H_u = ot.Uniform(990, 1110)
T_l = ot.Uniform(63.1, 116)
H_l = ot.Uniform(700, 820)
L = ot.Uniform(1120, 1680)
K_omega = ot.Uniform(9855, 12045)

#                  [r_omega,    r,    T_u,  H_u, T_l, H_l,    L, K_omega]
arg_max = np.array([[0.15,   100, 115600, 1110, 116, 700, 1120, 12045]])
arg_min = np.array([[0.05, 50000,  63070,  990,  63.1, 820, 1680,  9855]])

input_distrib = ot.JointDistribution([r_omega, r, T_u, H_u,
                                      T_l, H_l, L, K_omega])


def borehole(inputs):
    """ Borehole function. """
    r_omega0 = inputs[:, 0]
    r0 = inputs[:, 1]
    t_u0 = inputs[:, 2]
    h_u0 = inputs[:, 3]
    t_l0 = inputs[:, 4]
    h_l0 = inputs[:, 5]
    l0 = inputs[:, 6]
    k_omega0 = inputs[:, 7]

    res1 = 2*np.pi * t_u0 * (h_u0 - h_l0)
    res2 = np.log(r0/r_omega0)
    res3 = 1 + 2*l0*t_u0 / (res2*r_omega0**2*k_omega0) + t_u0/t_l0

    return res1/(res2*res3)


print(f'Maximum of output: {borehole(arg_max)}')
print(f'Minimum of output: {borehole(arg_min)}')

input_sample = np.array(input_distrib.getSample(SAMPLE_SIZE))
output_sample = borehole(input_sample)


with open(FILENAME, 'wb') as f:
    pickle.dump(output_sample, f)
