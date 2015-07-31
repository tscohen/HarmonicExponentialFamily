
import numpy as np
from scipy.fftpack import rfft, irfft

from HarmonicDensity import HarmonicDensity

class S1HarmonicDensity(HarmonicDensity):

    def __init__(self, L, oversampling_factor=2):
        """
        Harmonic density for the circle S^1.

        :param L: bandlimit. Sufficient statistics are cos(nx), sin(nx) for n=1, ..., L
        :param oversampling_factor: compute FFTs for bandlimit oversampling_factor * L
        """

        self.n_sufficient_statistics = 2 * L

        self.L = L
        self.even = np.arange(0, 2 * L, 2)
        self.odd = np.arange(1, 2 * L, 2)

        self.L_os = self.L * oversampling_factor
        self.even_os = np.arange(0, 2 * self.L_os, 2)
        self.odd_os = np.arange(1, 2 * self.L_os, 2)

    def analytical_moments(self, eta):
        """
        Compute the analytical moments for the S1 (circle) Harmonic Density, also known as the generalized von-Mises.

        :param eta: the natural parameters of the distribution.
        :return:
        """
        negative_energy = irfft(np.hstack([[0], eta]), n=self.L_os) * (self.L_os / 2.)
        maximum = np.max(negative_energy)
        unnormalized_moments = rfft(np.exp(negative_energy - maximum)) / (self.L_os / 2.) * np.pi
        moments = unnormalized_moments[1:eta.size + 1] / unnormalized_moments[0]
        lnz = np.log(unnormalized_moments[0]) + maximum
        return moments, lnz

    def empirical_moments(self, x, average=True):
        """
        Compute the empirical moments of the sample x

        :param x: dataset shape (N,)
        :return: the moments 1/N sum_i=1^N T(x_i)
        """
        empirical_moments = np.zeros(2 * self.L)
        empirical_moments[self.even] = np.sum(
            np.cos(np.arange(1, self.L + 1)[np.newaxis, :] * x[:, np.newaxis]), axis=0)
        empirical_moments[self.odd] = np.sum(
            np.sin(np.arange(1, self.L + 1)[np.newaxis, :] * x[:, np.newaxis]), axis=0)
        if average:
            return empirical_moments / x.shape[0]
        else:
            return empirical_moments