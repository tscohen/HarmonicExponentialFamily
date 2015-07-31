
import numpy as np
from scipy.optimize import fmin_cg, fmin_l_bfgs_b

class HarmonicDensity(object):
    """
    Base class for the harmonic densities.

    A harmonic density is an exponential family distribution over a (Lie) group or homogeneous space M, whose
    sufficient statistics are given by the "harmonic functions" on that space (i.e. matrix elements of irreducible
    unitary representations).

    This class implements the general functionality that is shared between every harmonic density, such as
    the computation of gradients and maximum likelihood estimation. The computation of empirical and analytical
    moments depends on the particular manifold over which the density is defined, and should be implemented in
    a subclass.
    """

    def empirical_moments(self, x, average=False):
        """
        Compute the empirical moments,
        E_data(T) = 1/N sum_{i=1}^N T(x^i)

        If average is False, we don't divide by N.

        :param x: input points on M
        :param average: if True, divide by number of data points, otherswise just sum.
        :return: the empirical moments
        """
        raise NotImplementedError('Not implemented in base class.')

    def analytical_moments(self, eta):
        """
        Compute the analytical moments,
        E_model(T) = int_M p(x) T(x) dx = int_M 1/Z exp(eta^T  T(x)) T(x) dx
        and the log-partition function:
        ln Z_eta = ln \int_M exp(eta^T T(x)) dx

        :param eta: the parameters of the model
        :return: the analytical moments and ln Z_eta.
        """
        raise NotImplementedError('Not implemented in base class.')

    def negative_energy(self, x, eta):
        """
        Compute the negative energy of x under the harmonic density with parameters eta.
        The negative energy is equal to the unnormalized log-probability:
         -E_eta(x) = ln p(x | eta) + ln Z_eta = eta^T T(x)

        :param x: array of points on M
        :param eta: array of natural parameter vectors. The last axis corresponds to the sufficient statistics of
        this distribution. The other axes should be broad-cast compatible with x.
        :return: the negative energy for each point in x and parameter-vector in eta, after broadcasting them together.
        """
        # The negative energy is just the dot product of eta and the empirical moments T(x).
        return np.einsum('...i,...i->...', eta, self.empirical_moments(x, average=False))

    def grad_log_p(self, eta, empirical_moments):
        """
        Compute the gradient of the average log-likelihood with respect to eta:
        grad_eta (1/N) sum_{n=1}^N log p(x_n | eta)
         =
        E[T(x)]_data - E[T(x)]_eta,
        where the expectations are with respect to the data distribution (empirical average) or the model distribution
        with eta determining the model. The empirical_moments = E[T(x)]_data can be computed by self.empirical_moments,
        while the analytical moments E[T(x)]_eta are computed by this function using self.moments.

        :param eta: the natural parameters of the harmonic density
        :param empirical_moments: the empirical moments of a dataset, as computed by self.empirical_moments
        :return: the gradient of the average log-likelihood with respect to eta.
        """
        moments, _ = self.analytical_moments(eta)
        return empirical_moments - moments

    def log_p_and_grad(self, eta, empirical_moments):
        """
        Compute the gradient of the log probability of the density given by eta,
        evaluated at a sample of data summarized by the empirical moments.
        The average log-prob is:
        1/N sum_i=1^N ln  p(x_i | eta)
         =
        1/N sum_i=1^N eta^T T(x_i) - ln Z_eta
         =
        eta^T T_bar - ln Z_eta
        where T_bar = 1/N sum_i=1^N T(x_i) are the empirical moments, as computed by self.empirical_moments(X).

        The gradient of the average log-prob is:
        T_bar - E_eta[T(x)]
        where E_eta[T(x)] are the moments of p(x|eta), as computed by self.moments(eta).

        :param eta: the natural parameters of the distribution
        :param empirical_moments: the average sufficient statistics, as computed by self.empirical_moments(X)
        :return: the gradient of the average log-prob with respect to eta, and the average log prob itself.
        """
        moments, lnz = self.analytical_moments(eta)
        grad_logp = empirical_moments - moments
        logp = eta.dot(empirical_moments) - lnz
        return logp, grad_logp

    def mle_lbfgs(self, empirical_moments, eta_init=None, sigma_inv=None, factr=1e12, pgtol=1e-4, verbose=False):
        """
        Perform maximum-likelihood estimation (MLE) using the L-BFGS optimizer.

        :param empirical_moments: the empirical moments of a dataset,
         as computed by self.empirical_moments(x, average=True)
        :param eta_init: where to initialize the optimization. If None (default), it is initialized to the null vector.
        :param sigma_inv: inverse of the diagonal covariance matrix of prior p(eta), used as a regularizer.
        :param factr: the L-BFGS iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps,
         where eps is the machine precision, which is automatically generated by the code.
         Typical values for factr are:
         1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        :param pgtol: the L-BFGS iteration will stop when max{|proj g_i | i = 1, ..., n} <= pgtol,
         where pg_i is the i-th component of the projected gradient.
        :param verbose: when set to True, this method will print diagnostic information.
        """

        # If no initialization for eta is provided, set eta_init to the null vector
        if eta_init is None:
            eta = np.zeros(self.n_sufficient_statistics)
        else:
            eta = eta_init.copy()

        # Define the objective_and_grad function to be used by the optimizer, either with or without regularization
        if sigma_inv is None:  # No regularization
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                return -logp, -grad
        else:
            # lnz_prior = 0.5 * SigmaInv.size * np.log(2 * np.pi) - 0.5 * np.sum(np.log(SigmaInv))
            def objective_and_grad(eta):
                logp, grad = self.log_p_and_grad(eta, empirical_moments)
                sigma_inv_eta = sigma_inv * eta
                logp += -0.5 * eta.dot(sigma_inv_eta)  # - lnz_prior
                grad += -sigma_inv_eta
                return -logp, -grad

        # Find the optimal parameters eta and the value of the objective at the optimum
        opt_eta, opt_neg_logp, info = fmin_l_bfgs_b(objective_and_grad, x0=eta, iprint=int(verbose) - 1,
                                                    factr=factr, pgtol=pgtol, maxiter=1000)

        # Finally, compute Z for the optimal eta:
        _, lnz = self.analytical_moments(opt_eta)

        # If desired, print diagnostics, then return
        if verbose:
            if sigma_inv is None:
                print 'Maximum log likelihood:', -opt_neg_logp
            else:
                print 'Maximum regularized log likelihood:', -opt_neg_logp
            print 'Optimization info:', info['warnflag'], info['task'], np.mean(info['grad'])

        return opt_eta, lnz

    def mle_cg(self, empirical_moments, eta_init=None, verbose=True):

        if eta_init is None:
            eta = np.zeros(self.n_sufficient_statistics)
        else:
            eta = eta_init.copy()

        def objective(eta):
            logp, _ = self.log_p_and_grad(eta, empirical_moments)
            return -logp

        def grad(eta):
            _, grad = self.log_p_and_grad(eta, empirical_moments)
            return -grad
        eta_min, logp_min, fun_calls, grad_calls, warnflag = fmin_cg(f=objective, fprime=grad, x0=eta, full_output=True)

        if verbose:
            print 'min log p:', logp_min
            print 'fun_calls:', fun_calls
            print 'grad_calls:', grad_calls
            print 'warnflag:', warnflag

        # Finally, compute Z:
        _, lnZ = self.analytical_moments(eta_min)
        return eta_min, lnZ

    def mle_sgd(self, empirical_moments, eta_init=None, learning_rate=0.1, max_iter=1000, verbose=True):

        if eta_init is None:
            eta = np.zeros(self.n_sufficient_statistics)
        else:
            eta = eta_init.copy()

        for i in range(max_iter):
            log_p, grad_log_p = self.log_p_and_grad(eta, empirical_moments)
            eta += learning_rate * grad_log_p
            if verbose:
                print 'log-prob:', log_p

        # Finally, compute Z:
        _, lnZ = self.analytical_moments(eta)
        return eta, lnZ
