
"""
There are many different conventions for the normalization and phase of the spherical harmonics.
This script provides functions that can evaluate the spherical harmonics for many of these conventions,
which should be useful for compatibility with various external libraries and mathematical texts.
"""

import numpy as np
from scipy.special import sph_harm
from scipy.misc import factorial


def sh(l, m, theta, phi, field='real', normalization='quantum', condon_shortley=True):
    """
    Compute spherical harmonics Y_lm(theta, phi) for various definitions of Y_lm.

    :param l: the degree of the spherical harmonics
    :param m: the order of the spherical harmonics; must satisfy -l <= m <= l
    :param theta: first spherical coordinate in [0, pi)
    :param phi: second spherical coordinate in [0, 2pi)
    :param field: 'real' or 'complex'
    :param normalization: how to normalize the SH:
    'seismology', 'quantum', 'geodesy', 'unnormalized', 'nfft'
    :param condon_shortley: whether or not to use Condon-Shortley phase convention.
    :return: the value of Y_lm(theta, phi)
    """
    if field == 'real':
        return rsh(l, m, theta, phi, normalization, condon_shortley)
    elif field == 'complex':
        return csh(l, m, theta, phi, normalization, condon_shortley)
    else:
        raise ValueError('Unknown field: ' + str(field))


def sh_batch(L_max, theta, phi):
    """
    Compute all spherical harmonics up to and including degree L_max, for angles theta and phi.

    This function is currently rather hacky and should be refactored,
    but the method used here is very fast and numerically stable, compared to builtin scipy functions.

    :param L_max:
    :param theta:
    :param phi:
    :return:
    """

    from ..SO3.pinchon_hoggan.pinchon_hoggan import apply_rotation_block, make_c2b
    from ..SO3.irrep_bases import change_of_basis_function

    # Compute the values of l and m as they range from (0, 0) to (L_max, L_max)
    irreps = np.arange(L_max + 1)
    ls = [[ls] * (2 * ls + 1) for ls in irreps]
    ls = np.array([ll for sublist in ls for ll in sublist])  # 1, 1, 1, 2, 2, 2, 2, 2, ...
    ms = [range(-ls, ls + 1) for ls in irreps]
    ms = np.array([mm for sublist in ms for mm in sublist])  # -1, 0, 1, -2, -1, 0, 1, 2, ...

    # Get a vector Y that selects the 0-frequency component from each irrep in the centered basis
    # If D = D(theta, phi, 0) is a Wigner D matrix,
    # then D Y is the center column of D, which is equal to the spherical harmonics.
    Y = (ms == 0).astype(float)

    # Change to / from the block basis (since the rotation code works in that basis)
    c2b = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'centered', 'cs'),
                                   to=('real', 'quantum', 'block', 'cs'))
    b2c = change_of_basis_function(irreps,
                                   frm=('real', 'quantum', 'block', 'cs'),
                                   to=('real', 'quantum', 'centered', 'cs'))
    Yb = c2b(Y)

    # Rotate Yb:
    # TODO: move this out of here
    import os
    # J_block = np.load(os.path.join(os.path.dirname(__file__), '../SO3/pinchon_hoggan', 'J_block_0-278.npy'))
    J_block = np.load(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SO3/pinchon_hoggan', 'J_block_0-278.npy')))
    J_block = list(J_block[irreps])

    g = np.zeros((theta.size, 3))
    g[:, 0] = phi
    g[:, 1] = theta
    c2b = make_c2b(irreps)
    TYb = apply_rotation_block(g=g, X=Yb[np.newaxis, :],
                               irreps=irreps, c2b=c2b,
                               J_block=J_block, l_max=np.max(irreps))

    # Change back to centered basis
    TYc = b2c(TYb.T).T  # b2c doesn't work properly for matrices, so do a transpose hack

    # The SH obtained so far are equal to (real, nfft, centered, cs) spherical harmonics
    # Change to real quantum centered cs
    c = change_of_basis_function(irreps,
                                 frm=('real', 'nfft', 'centered', 'cs'),
                                 to=('real', 'quantum', 'centered', 'cs'))
    TYc2 = c(TYc)

    return TYc2

def rsh(l, m, theta, phi, normalization='quantum', condon_shortley=True):
    """
    Compute the real spherical harmonic (RSH) S_l^m(theta, phi).

    The RSH are obtained from Complex Spherical Harmonics (CSH) as follows:
    if m < 0:
        S_l^m = i / sqrt(2) * (Y_l^m - (-1)^m Y_l^{-m})
    if m == 0:
        S_l^m = Y_l^0
    if m > 0:
        S_l^m = 1 / sqrt(2) * (Y_l^{-m} + (-1)^m Y_l^m)
     (see [1])

    Various normalizations for the CSH exist, see the CSH() function. Since the CSH->RSH change of basis is unitary,
    the orthogonality and normalization properties of the RSH are the same as those of the CSH from which they were
    obtained. Furthermore, the operation of changing normalization and that of changeing field
    (complex->real or vice-versa) commute, because the ratio c_m of normalization constants are always the same for
    m and -m (to see this that this implies commutativity, substitute Y_l^m * c_m for Y_l^m in the above formula).

    Pinchon & Hoggan [2] define a different change of basis for CSH -> RSH, but they also use an unusual definition
    of CSH. To obtain RSH as defined by Pinchon-Hoggan, use this function with normalization='quantum'.

    References:
    [1] http://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    [2] Rotation matrices for real spherical harmonics: general rotations of atomic orbitals in space-fixed axes.

    :param l: non-negative integer; the degree of the CSH.
    :param m: integer, -l <= m <= l; the order of the CSH.
    :param theta: the colatitude / polar angle,
    ranging from 0 (North Pole, (X,Y,Z)=(0,0,1)) to pi (South Pole, (X,Y,Z)=(0,0,-1)).
    :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    :param normalization: how to normalize the RSH:
    'seismology', 'quantum', 'geodesy', 'unnormalized', 'nfft'
    these are immediately passed to the CSH functions, and since the change of basis
    from CSH to RSH is unitary, the orthogonality and normalization properties of these conventions are the same as
    those of the corresponding CSHs.
    :return: the value of the real spherical harmonic S^l_m(theta, phi)
    """
    l, m, theta, phi = np.broadcast_arrays(l, m, theta, phi)
    # Get the CSH for m and -m, using Condon-Shortley phase (regardless of whhether CS is requested or not)
    # The reason is that the code that changes from CSH to RSH assumes CS phase.
    a = csh(l=l, m=m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)
    b = csh(l=l, m=-m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)

    y = ((m > 0) * np.array((b + ((-1)**m) * a).real / np.sqrt(2.))
         + (m < 0) * np.array((1j * a - 1j * ((-1)**(-m)) * b).real / np.sqrt(2.))
         + (m == 0) * np.array(a.real))

    if condon_shortley:
        return y
    else:
        # Cancel the CS phase of y (i.e. multiply by -1 when m is both odd and greater than 0)
        return y * ((-1) ** (m * (m > 0)))


def csh(l, m, theta, phi, normalization='quantum', condon_shortley=True):
    """
    Compute Complex Spherical Harmonics (CSH) Y_l^m(theta, phi).
    Unlike the scipy.special.sph_harm function, we use the common convention that
    theta is the polar angle (0 to pi) and phi is the azimuthal angle (0 to 2pi).

    The spherical harmonic 'backbone' is:
    Y_l^m(theta, phi) = P_l^m(cos(theta)) exp(i m phi)
    where P_l^m is the associated Legendre function as defined in the scipy library (scipy.special.sph_harm).

    Various normalization factors can be multiplied with this function.
    -> seismology: sqrt( ((2 l + 1) * (l - m)!) / (4 pi * (l + m)!) )
    -> quantum: (-1)^2 sqrt( ((2 l + 1) * (l - m)!) / (4 pi * (l + m)!) )
    -> unnormalized: 1
    -> geodesy: sqrt( ((2 l + 1) * (l - m)!) / (l + m)! )
    -> nfft: sqrt( (l - m)! / (l + m)! )

    The 'quantum' and 'seismology' CSH are normalized so that
    <Y_l^m, Y_l'^m'>
    =
    int_S^2 Y_l^m(theta, phi) Y_l'^m'* dOmega
    =
    delta(l, l') delta(m, m')
    where dOmega is the volume element for the sphere S^2:
    dOmega = sin(theta) dtheta dphi
    The 'geodesy' convention have unit power, meaning the norm is equal to the surface area of the unit sphere (4 pi)
    <Y_l^m, Y_l'^m'> = 4pi delta(l, l') delta(m, m')

    On each of these normalizations, one can optionally include a Condon-Shortley phase factor:
    (-1)^m   (if m > 0)
    1        (otherwise)
    Note that this is the definition of Condon-Shortley according to wikipedia [1], but other sources call a
    phase factor of (-1)^m a Condon-Shortley phase (without mentioning the condition m > 0).

    References:
    [1] http://en.wikipedia.org/wiki/Spherical_harmonics#Conventions

    :param l: non-negative integer; the degree of the CSH.
    :param m: integer, -l <= m <= l; the order of the CSH.
    :param theta: the colatitude / polar angle,
    ranging from 0 (North Pole, (X,Y,Z)=(0,0,1)) to pi (South Pole, (X,Y,Z)=(0,0,-1)).
    :param phi: the longitude / azimuthal angle, ranging from 0 to 2 pi.
    :param normalization: how to normalize the CSH:
    'seismology', 'quantum', 'geodesy', 'unnormalized', 'nfft'.
    :return: the value of the complex spherical harmonic Y^l_m(theta, phi)
    """
    if normalization == 'quantum':
        # y = ((-1) ** m) * sph_harm(m, l, theta=phi, phi=theta)
        y = ((-1) ** m) * sph_harm(m, l, phi, theta)
    elif normalization == 'seismology':
        # y = sph_harm(m, l, theta=phi, phi=theta)
        y = sph_harm(m, l, phi, theta)
    elif normalization == 'geodesy':
        # y = np.sqrt(4 * np.pi) * sph_harm(m, l, theta=phi, phi=theta)
        y = np.sqrt(4 * np.pi) * sph_harm(m, l, phi, theta)
    elif normalization == 'unnormalized':
        # y = sph_harm(m, l, theta=phi, phi=theta) / np.sqrt((2 * l + 1) * factorial(l - m) /
        #                                                    (4 * np.pi * factorial(l + m)))
        y = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) * factorial(l - m) /
                                                           (4 * np.pi * factorial(l + m)))
    elif normalization == 'nfft':
        # y = sph_harm(m, l, theta=phi, phi=theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
        y = sph_harm(m, l, phi, theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        raise ValueError('Unknown normalization convention:' + str(normalization))

    if condon_shortley:
        # The sph_harm function already includes CS phase
        return y
    else:
        # Cancel the CS phase in sph_harm (i.e. multiply by -1 when m is both odd and greater than 0)
        return y * ((-1) ** (m * (m > 0)))