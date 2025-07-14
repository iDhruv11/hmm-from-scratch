import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    def __init__(self, n_states, n_dim, max_iter=50, tol=1e-4, seed=None):
        self.n = n_states
        self.d = n_dim
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        self.pi = self._norm(self.rng.random(self.n))
        self.A = self._norm(self.rng.random((self.n, self.n)), axis=1)
        self.means = self.rng.normal(0, 1, (self.n, self.d))
        self.covs = np.array([np.eye(self.d) for _ in range(self.n)])

    def _norm(self, x, axis=None):
        return x / x.sum(axis=axis, keepdims=True)

    def _logpdf(self, x, mu, cov):
        d = self.d
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = x - mu
        return -0.5 * (d * np.log(2 * np.pi) + np.log(det) + diff.T @ inv @ diff)

    def _log_emissions(self, obs):
        T = len(obs)
        logB = np.zeros((T, self.n))
        for t in range(T):
            for i in range(self.n):
                logB[t, i] = self._logpdf(obs[t], self.means[i], self.covs[i])
        return logB

    def _forward(self, logB):
        T = logB.shape[0]
        la = np.zeros((T, self.n))
        la[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            for j in range(self.n):
                la[t, j] = logB[t, j] + logsumexp(la[t-1] + np.log(self.A[:, j]))
        return la

    def _backward(self, logB):
        T = logB.shape[0]
        lb = np.zeros((T, self.n))
        for t in range(T-2, -1, -1):
            for i in range(self.n):
                lb[t, i] = logsumexp(np.log(self.A[i]) + logB[t+1] + lb[t+1])
        return lb

    def _e_step(self, obs):
        logB = self._log_emissions(obs)
        la = self._forward(logB)
        lb = self._backward(logB)

        log_g = la + lb
        log_g -= logsumexp(log_g, axis=1, keepdims=True)
        gamma = np.exp(log_g)

        T = len(obs)
        xi = np.zeros((T-1, self.n, self.n))
        for t in range(T-1):
            lx = (la[t][:, None] + np.log(self.A) + logB[t+1][None, :] + lb[t+1][None, :])
            lx -= logsumexp(lx)
            xi[t] = np.exp(lx)

        loglik = logsumexp(la[-1])
        return gamma, xi, loglik

    def _m_step(self, obs, gamma, xi):
        self.pi = gamma[0]
        self.A = xi.sum(axis=0)
        self.A = self._norm(self.A, axis=1)
        for i in range(self.n):
            w = gamma[:, i][:, None]
            mu = (w * obs).sum(axis=0) / w.sum()
            diff = obs - mu
            cov = (w[:, :, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / w.sum()
            self.means[i] = mu
            self.covs[i] = cov + 1e-6 * np.eye(self.d)

    def fit(self, obs):
        prev_ll = -np.inf
        for it in range(self.max_iter):
            gamma, xi, ll = self._e_step(obs)
            self._m_step(obs, gamma, xi)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            print(f"Iter {it}, loglik={ll:.3f}")

    def decode(self, obs):
        logB = self._log_emissions(obs)
        T = len(obs)
        delta = np.zeros((T, self.n))
        psi = np.zeros((T, self.n), dtype=int)

        delta[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            for j in range(self.n):
                seq_probs = delta[t-1] + np.log(self.A[:, j])
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = seq_probs[psi[t, j]] + logB[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in reversed(range(T-1)):
            states[t] = psi[t+1, states[t+1]]
        return states
