T.S. Cohen, M. Welling, Harmonic Exponential Families on Manifolds. Proceedings of the International Conference on Machine Learning (ICML), 2015

[`pdf <https://tacocohen.files.wordpress.com/2015/05/hef.pdf>`_] [`supp. mat. <https://tacocohen.files.wordpress.com/2015/05/hef_supplementary_material.pdf>`_]


Setup
=====

To compile the Cython modules, run:

python setup.py build_ext --inplace

To use the NFSFT (Non-equispaced Fast Spherical Fourier Transform), install my fork of the PyNFFT bindings, which supports the NFSFT (https://github.com/tscohen/pyNFFT).
When compiling the NFFT library, make sure NFSFT support is enabled.
