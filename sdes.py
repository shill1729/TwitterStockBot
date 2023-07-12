from emas import *
from optport import mv_solver


class MultiGbm:

    def __init__(self, drift=None, Sigma=None):
        """ Multivariate geometric Brownian motion with drift vector and covariance of log-returns
        """
        self.drift = drift
        self.Sigma = Sigma

    def __str__(self):
        return "Drift vector = " + str(self.drift) + "\n Covariance = " + str(self.Sigma)

    def fit(self, X, ema_filter=0.0, timescale=1.0 / 252.0):
        """ Estimate drift vector and covariance matrix of log-returns using either
        a naive sample estimate or exponentially weighted averages.
        """
        # Assuming X is a pandas series of log-returns
        # Fit a GBM model with naive estimators
        if ema_filter == 0.0:
            # print("Using naive-estimates")
            self.Sigma = np.cov(X, rowvar=False) / timescale
            self.drift = np.mean(X, axis=0) / timescale + 0.5 * np.diagonal(self.Sigma)
            self.drift = self.drift.values
        if ema_filter > 0.0:
            # print("Using EMA filter")
            self.Sigma = ewmc(X, ema_filter) / timescale
            self.drift = ema(X, ema_filter) / timescale + 0.5 * np.diagonal(self.Sigma)
            self.Sigma = self.Sigma.values
        elif ema_filter < 0:
            raise ValueError("'ema_filter' must be non-negative")
        return None

    def kelly_criterion(self, r=0):
        """ Compute log-optimal portfolio allocations
        """
        # print(mv_solver(self.drift - r, self.Sigma))
        # print(kc(self.drift-r,self.Sigma))
        return mv_solver(self.drift - r, self.Sigma)
        # return kc(self.drift-r,self.Sigma)

    def min_variance(self):
        """ Compute the minimum variance portfolio
        """
        d = self.Sigma.shape[0]
        return mv_solver(np.zeros(d), self.Sigma)
