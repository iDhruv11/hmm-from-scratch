
import numpy as np


def safe_normalize_rows(mat, eps=1e-12):
    mat = np.array(mat, dtype=float)
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = eps
    return mat / sums


class HMM:
    def __init__(self, pi, A, B):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        assert self.pi.ndim == 1
        assert self.A.shape[0] == self.A.shape[1] == self.pi.shape[0]
        assert self.B.shape[0] == self.pi.shape[0]
        self.pi = self.pi / self.pi.sum()
        self.A = safe_normalize_rows(self.A)
        self.B = safe_normalize_rows(self.B)

    def forward(self, obs):
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        N = self.pi.shape[0]
        fwd = np.zeros((T, N))
        scale = np.zeros(T)

        fwd[0] = self.pi * self.B[:, obs[0]]
        scale[0] = fwd[0].sum()
        if scale[0] == 0:
            scale[0] = 1e-300
        fwd[0] /= scale[0]

        for t in range(1, T):
            pred = fwd[t - 1] @ self.A
            fwd[t] = pred * self.B[:, obs[t]]
            scale[t] = fwd[t].sum()
            if scale[t] == 0:
                scale[t] = 1e-300
            fwd[t] /= scale[t]

        ll = -np.sum(np.log(scale))
        return fwd, scale, ll

    def backward(self, obs, scale):
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        N = self.pi.shape[0]
        bwd = np.zeros((T, N))

        bwd[T - 1] = 1.0 / scale[T - 1]
        for t in range(T - 2, -1, -1):
            tmp = (self.B[:, obs[t + 1]] * bwd[t + 1])
            bwd[t] = (self.A * tmp[None, :]).sum(axis=1)
            bwd[t] /= scale[t]
        return bwd

    def posterior(self, obs):
        fwd, scale, ll = self.forward(obs)
        bwd = self.backward(obs, scale)
        T, N = fwd.shape

        gamma = fwd * bwd
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            numer = (fwd[t][:, None] * self.A) * (self.B[:, obs[t + 1]] * bwd[t + 1])[None, :]
            denom = numer.sum()
            if denom == 0:
                denom = 1e-300
            xi[t] = numer / denom
        return gamma, xi, ll

    def viterbi(self, obs):
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        N = self.pi.shape[0]

        logA = np.log(self.A + 1e-300)
        logB = np.log(self.B + 1e-300)
        logpi = np.log(self.pi + 1e-300)

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = logpi + logB[:, obs[0]]
        for t in range(1, T):
            for j in range(N):
                scores = delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + logB[j, obs[t]]

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        best_score = float(np.max(delta[T - 1]))
        return path, best_score

    def train(self, seqs, max_iter=50, tol=1e-6, verbose=False, restarts=1, seed=None):
        rng = np.random.default_rng(seed)
        best_model = None
        best_ll = -np.inf
        best_hist = None

        for r in range(restarts):
            if r == 0:
                pi = self.pi.copy()
                A = self.A.copy()
                B = self.B.copy()
            else:
                N = self.pi.shape[0]
                M = self.B.shape[1]
                pi = rng.random(N)
                pi = pi / pi.sum()
                A = rng.random((N, N))
                A = safe_normalize_rows(A)
                B = rng.random((N, M))
                B = safe_normalize_rows(B)

            candidate = HMM(pi, A, B)
            hist = []

            for it in range(max_iter):
                N = candidate.pi.shape[0]
                M = candidate.B.shape[1]

                pi_new = np.zeros(N)
                A_num = np.zeros((N, N))
                A_den = np.zeros(N)
                B_num = np.zeros((N, M))
                B_den = np.zeros(N)
                total_ll = 0.0

                for seq in seqs:
                    gamma, xi, ll = candidate.posterior(seq)
                    total_ll += ll

                    T = len(seq)
                    pi_new += gamma[0]

                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)

                    for t, o in enumerate(seq):
                        B_num[:, o] += gamma[t]
                    B_den += gamma.sum(axis=0)

                candidate.pi = pi_new / pi_new.sum()
                A_den_safe = A_den.copy()
                A_den_safe[A_den_safe == 0] = 1e-300
                candidate.A = A_num / A_den_safe[:, None]
                candidate.A = safe_normalize_rows(candidate.A)
                B_den_safe = B_den.copy()
                B_den_safe[B_den_safe == 0] = 1e-300
                candidate.B = B_num / B_den_safe[:, None]
                candidate.B = safe_normalize_rows(candidate.B)

                hist.append(total_ll)
                if verbose:
                    print(f"[restart {r+1}/{restarts}] iter {it+1} ll={total_ll:.6f}")

                if it > 0 and abs(hist[-1] - hist[-2]) < tol:
                    if verbose:
                        print("Converged.")
                    break

            if total_ll > best_ll:
                best_ll = total_ll
                best_model = candidate
                best_hist = hist

        if best_model is not None:
            self.pi = best_model.pi
            self.A = best_model.A
            self.B = best_model.B

        return best_ll, best_hist

    def score(self, seqs):
        total = 0.0
        for seq in seqs:
            _, _, ll = self.forward(seq)
            total += ll
        return total


def sample_hmm(model, T, seed=None):
    rng = np.random.default_rng(seed)
    N = model.pi.shape[0]
    M = model.B.shape[1]
    states = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)
    states[0] = rng.choice(N, p=model.pi)
    obs[0] = rng.choice(M, p=model.B[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(N, p=model.A[states[t - 1]])
        obs[t] = rng.choice(M, p=model.B[states[t]])
    return states, obs


def _demo():
    rng = np.random.default_rng(42)

    pi_true = np.array([0.6, 0.4])
    A_true = np.array([[0.7, 0.3],
                       [0.4, 0.6]])
    B_true = np.array([[0.5, 0.4, 0.1],
                       [0.1, 0.3, 0.6]])
    true_model = HMM(pi_true, A_true, B_true)

    n_seqs = 8
    seq_len = 200
    obs_list = []
    states_list = []
    for _ in range(n_seqs):
        s, o = sample_hmm(true_model, seq_len, seed=rng.integers(2**31 - 1))
        obs_list.append(o.tolist())
        states_list.append(s)

    print(f"Sampled {n_seqs} sequences of length {seq_len} (total tokens={n_seqs*seq_len})")

    N = 2
    M = 3
    pi_init = rng.random(N)
    pi_init /= pi_init.sum()
    A_init = rng.random((N, N)); A_init = safe_normalize_rows(A_init)
    B_init = rng.random((N, M)); B_init = safe_normalize_rows(B_init)

    model = HMM(pi_init, A_init, B_init)
    print("\nInitial random parameters:")
    print("pi:", model.pi)
    print("A:\n", model.A)
    print("B:\n", model.B)

    ll_before = model.score(obs_list)
    print(f"\nInitial total log-likelihood (all seqs): {ll_before:.6f}")

    print("\nTraining with Baum-Welch (restarts=5)...")
    best_ll, hist = model.train(obs_list, max_iter=200, tol=1e-6, verbose=False, restarts=5, seed=0)
    print(f"Best total log-likelihood after training: {best_ll:.6f}")

    print("\nLearned parameters:")
    print("pi:", model.pi)
    print("A:\n", model.A)
    print("B:\n", model.B)

    ll_true = true_model.score(obs_list)
    print(f"\nTrue HMM total log-likelihood (for same seqs): {ll_true:.6f}")

    path_vit, vll = model.viterbi(obs_list[0])
    print("\nViterbi path (first seq) first 30 states:", path_vit[:30])

    gamma, xi, seq_ll = model.posterior(obs_list[0])
    print("\nPosterior gamma (first sequence) first 6 timesteps:\n", gamma[:6])

    if hist:
        print("\nTraining log-likelihood history (last 10 iters):")
        print(hist[-10:])


if __name__ == "__main__":
    _demo()
