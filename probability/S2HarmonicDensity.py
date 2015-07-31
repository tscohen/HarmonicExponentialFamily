import numpy as np

from HarmonicDensity import HarmonicDensity
from ..spectral.S2_FFT_NFFT import S2FFT_NFFT
from ..grids import S2
from ..representations.S2.spherical_harmonics import sh, sh_batch


class S2HarmonicDensity(HarmonicDensity):

    def __init__(self, L, oversampling_factor=2, fft=None):

        self.L = L
        self.L_os = self.L * oversampling_factor
        self.n_sufficient_statistics = (self.L + 1) ** 2 - 1

        # Create arrays containing the (l, m) coordinates
        l = [[l] * (2 * l + 1) for l in range(1, self.L + 1)]
        self.ls = np.array([ll for sublist in l for ll in sublist])  # 1, 1, 1, 2, 2, 2, 2, 2, ...
        l_oversampled = [[l] * (2 * l + 1) for l in range(1, self.L_os + 1)]
        self.ls_oversampled = np.array([ll for sublist in l_oversampled for ll in sublist])

        m = [range(-l, l + 1) for l in range(1, self.L + 1)]
        self.ms = np.array([mm for sublist in m for mm in sublist])  # -1, 0, 1, -2, -1, 0, 1, 2, ...
        m_oversampled = [range(-l, l + 1) for l in range(1, self.L_os + 1)]
        self.ms_oversampled = np.array([mm for sublist in m_oversampled for mm in sublist])

        # Create a spherical FFT instance if it was not provided
        if fft is None:
            # Setup a spherical grid and corresponding quadrature weights, then instantiate the FFT
            convention = 'Clenshaw-Curtis'  # 'Gauss-Legendre'
            x = S2.meshgrid(b=self.L_os, convention=convention)
            w = S2.quadrature_weights(b=self.L_os, convention=convention)
            self.fft = S2FFT_NFFT(L_max=self.L_os, x=x, w=w)
        else:
            if fft.L_max < self.L_os:
                raise ValueError('fft.L must be larger than or equal to L * oversampling_factor')
            self.fft = fft

    def empirical_moments(self, X, average=True):
        """
        Compute the empirical moments of the sample X

        :param X: dataset shape N + (2,) where the last axis corresponds to the 2 spherical coordinates (theta, phi).
        :return: the moments 1/N sum_i=1^N T(x_i)
        """
        # TODO: use the sh_batch function of the spherical_harmonics script.
        # and then transforming by D(theta, phi, 0) or something similar. This matrix vector-multiplication can be
        # done efficiently by the Pinchon-Hoggan method. (or asymptotically even faster using other methods)
        T = sh(self.ls[np.newaxis, :], self.ms[np.newaxis, :],
               X[:, 0][:, np.newaxis], X[:, 1][:, np.newaxis],
               field='real', normalization='quantum', condon_shortley=True)

        # Compute all sufficient statistics up to self.L_max using an efficient batch method
        # The first element is discarded because it corresponds to Y_0^0(t, p) = const.
        # T = sh_batch(L_max=self.L_max, theta=X[:, 0], phi=X[:, 1])[..., 1:]

        if average:
            return T.mean(axis=0)
        else:
            return T

    def analytical_moments(self, eta):
        """
        Compute the analytical moments of the distribution determined by eta

        :param eta: array of natural parameter vectors
        :return: array of moment vectors
        """
        #
        eta_os = np.zeros((self.L_os + 1) ** 2)
        eta_os[1:eta.size + 1] = eta

        negative_energy = self.fft.synthesize(eta_os)
        maximum = np.max(negative_energy)
        unnormalized_moments = self.fft.analyze(np.exp(negative_energy - maximum))
        unnormalized_moments[0] *= np.sqrt(4 * np.pi)
        moments = unnormalized_moments / unnormalized_moments[0]
        lnZ = np.log(unnormalized_moments[0]) + maximum

        return moments[1:(self.L + 1) ** 2], lnZ