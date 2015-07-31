
GrouPy is a python library for numerical computations involving (Lie) groups and homogeneous spaces.

Setup
=====

To compile the Cython modules, run:

python setup.py build_ext --inplace

To use the NFSFT (Non-equispaced Fast Spherical Fourier Transform), install my fork of the PyNFFT bindings, which supports the NFSFT (https://github.com/tscohen/pyNFFT).
When compiling the NFFT library, make sure NFSFT support is enabled.
