
import numpy as np


# TODO: add unit-test to check moments agains numerical integration (have done this in terminal)

def _moments_numint(self, eta):
    """
    Compute the moments of p(x | eta) using numerical integration.
    This is very slow, and should only be used for testing purposes.

    TODO: move this to a unittest?

    :param eta:
    :return:
    """

    moments = np.zeros((self.L_max + 1) ** 2)

    f = lambda th, ph: np.exp(self.negative_energy([th, ph], eta))
    moments[0] = S2.integrate(f, normalize=False)

    for l in range(1, self.L_max + 1):
        for m in range(-l, l + 1):
            print 'integrating', l, m
            f = lambda th, ph: np.exp(self.negative_energy([th, ph], eta)) * sh(l, m, th, ph,
                                                    field='real', normalization='quantum', condon_shortley=True)
            moments[l ** 2 + l + m] = S2.integrate(f, normalize=False)

    return moments[1:] / moments[0], moments[0]

def _moment_numerical_integration(self, eta, l, m):
    """
    Compute the (l,m)-moment of the density with natural parameter eta using slow numerical integration.
    The output of this function should be equal to the *unnormalized* moment as it comes out of the FFT
    (without dividing by Z).

    :param eta:
    :param l:
    :param m:
    :return:
    """
    f = lambda theta, phi: (np.exp(self.negative_energy(np.array([[theta, phi]]), eta))
                            * sh(l, m, theta, phi,
                                 field='real', normalization='quantum',
                                 condon_shortley=True))
    return S2.integrate(f)

