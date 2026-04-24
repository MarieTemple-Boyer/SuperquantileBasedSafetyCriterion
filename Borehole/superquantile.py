""" Numerical estimation of superquantile. """

import numpy as np

def superquantile(sample, alpha):
    """ Compute the alpha-quantile using the minimisation formula
        (and by sorting the sample).
    >>> superquantile(SAMPLE, alpha=0.8)
    np.float64(6.0)
    >>> superquantile(SAMPLE, alpha=0)
    np.float64(3.2)
    >>> superquantile(SAMPLE, alpha=1)
    np.float64(6.0)
    >>> superquantile(SAMPLE_BIG, alpha=0.95)
    np.float64(2.0466469703496997)
    """
    if alpha == 0:
        return np.mean(sample)

    sample = np.sort(sample)
    if alpha == 1:
        return sample[-1]

    # Compute the minimum of phi_n(sample) where
    #   phi_n(c) = c + 1/(1-alpha) * mean((sample-c)_+)
    def phi_n(c):
        return c + 1/(1-alpha) * np.mean(np.maximum(0, sample-c))
    return phi_n(sample[int(np.ceil(alpha*len(sample)))-1])

if __name__ == "__main__":
    import doctest
    import openturns as ot
    ot.RandomGenerator.SetSeed(0)
    SAMPLE = ot.Sample.BuildFromPoint([5, 3, 6, -1, 3])
    SAMPLE = np.array(SAMPLE)[:,0]
    SAMPLE_BIG = ot.Normal(0, 1).getSample(2*10**4)
    SAMPLE_BIG = np.array(SAMPLE_BIG)[:,0]
    doctest.testmod()
