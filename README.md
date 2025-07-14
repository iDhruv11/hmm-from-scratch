# Hidden Markov Models from Scratch - Discrete and Gaussian Extensions

- `hmm.py`: A from-scratch, NumPy-only discrete Hidden Markov Model (HMM) with scaled Forward-Backward, Viterbi, posterior marginals (gamma, xi), Baum-Welch training over multiple sequences, and sampling.
- `extended.py`: A Gaussian-emission HMM (continuous HMM) implemented in log-space with `logsumexp`, supporting multivariate normal emissions with state-specific means and covariances.

As an NLP enthusiast, I wanted to internalize the classical probabilistic modeling of sequences - how we modeled text and speech before transformers. HMMs offer a clean, interpretable framework to learn:

- How latent state dynamics and emission models interact
- How dynamic programming (Forward-Backward, Viterbi) makes inference tractable
- How EM/maximum-likelihood training works in practice with real numerical stability concerns

## Difference between `hmm.py` to `extended.py`?

- Discrete to continuous emissions: `hmm.py` uses categorical emissions (each state has a probability over symbols). `extended.py` models real-valued vectors with multivariate Gaussians (mean + covariance per state).
- Numerics: `extended.py` computes everything in log-space using `scipy.special.logsumexp` for stability on long sequences. `hmm.py` uses classic scaling factors for stability in the discrete case.
- Expressiveness: Gaussian emissions let the model capture continuous features (e.g., MFCCs for speech, learned embeddings, sensor data) that categorical HMMs cannot represent well.

## Quick start

### Environment

- Python 3.9+
- NumPy
- SciPy (for `extended.py`)

Install dependencies (SciPy pulls NumPy automatically):

```bash
pip install numpy scipy
```

### Discrete HMM demo (`hmm.py`)

`hmm.py` includes a runnable demo that:
- Samples synthetic sequences from a known HMM
- Trains a fresh HMM with Baum-Welch with restarts
- Prints learned parameters, total log-likelihood, posterior marginals, and a Viterbi path

Run it:

```bash
python3 hmm.py
```

Expected behavior: it should print initial/random params, improve total log-likelihood over EM, and show example posteriors and a Viterbi path for a sample sequence.

### Gaussian HMM usage (`extended.py`)

`extended.py` provides a `GaussianHMM` you can import. Example usage:

```python
import numpy as np
from extended import GaussianHMM

# Toy 2-state, 2D data
T, n_states, n_dims = 300, 2, 2
rng = np.random.default_rng(0)

# Generate a simple dataset (unlabeled) around two clusters
obs = np.vstack([
    rng.normal([-2, 0], [0.7, 0.7], size=(T//2, n_dims)),
    rng.normal([+2, 0], [0.7, 0.7], size=(T//2, n_dims)),
])

hmm = GaussianHMM(n_states=n_states, n_dims=n_dims, max_iter=50, tol=1e-4, random_state=0)

# Fit and decode
hmm.fit(obs)
states = hmm.decode(obs)
print("means=\n", hmm.means)
print("first 20 states:", states[:20])
```

Notes:
- Inputs are real-valued vectors of shape `(T, n_dims)`.
- The implementation uses full covariance matrices per state with a tiny diagonal regularizer.

## Feature overview

### `hmm.py` (Discrete HMM)
- Scaled Forward-Backward for numerical stability
- Posterior marginals (gamma) and pairwise posteriors (xi)
- Viterbi decoding (log-space)
- Stable log-likelihood via scaling factors
- Baum-Welch EM across multiple sequences with restarts
- Sampling utility and end-to-end demo

### `extended.py` (Gaussian HMM)
- Multivariate Gaussian emissions per state (means + covariances)
- Log-space Forward-Backward with `logsumexp`
- EM updates for means, covariances, transitions, and initial distribution
- Viterbi decoding in log-space

## What I learned building these

- Implementing Forward-Backward twice: with per-step scaling (discrete) and in log-space (Gaussian), and why both avoid underflow.
- EM in practice: accumulating expected counts (xi, gamma), handling zero counts, and ensuring row-stochastic constraints during the M-step.
- Numerical stability: clipping/epsilons, log-sum-exp, covariance regularization, and how tiny mistakes destabilize EM.
- Viterbi vs. posteriors: decoding the single MAP path versus computing full marginals, and when each is appropriate.
- Modeling choices matter: discrete emissions are simple and fast; Gaussian emissions unlock continuous data but require matrix algebra (inverses, determinants) and care with ill-conditioned covariances.
