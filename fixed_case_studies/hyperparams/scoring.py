"""Pure scoring helpers for the hyperparameter-selection workflow.

Everything in this module is deterministic and unit-testable (no fitting):
pointwise-elpd aggregation, PSIS-LOO comparison tables with difference
standard errors, pseudo-BMA / stacking weights, and the empirical CRPS.

Formulas follow Vehtari, Gelman & Gabry (2017), "Practical Bayesian model
evaluation using leave-one-out cross-validation and WAIC" (Stat. Comput. 27):
- ``se(elpd) = sqrt(n * var(elpd_i))`` (sample variance, ddof=1);
- ``se(diff) = sqrt(n * var(elpd_i^A - elpd_i^B))``;
- pseudo-BMA weights with a Bayesian bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pointwise elpd
# ---------------------------------------------------------------------------


def pointwise_elpd(model) -> np.ndarray:
    """Extract the pointwise PSIS-LOO elpd of a fitted model.

    ``model`` must expose ``.trace`` with a ``log_likelihood`` group (i.e.
    ``model.compute_log_likelihood()`` was called).  Returns the pointwise
    elpd as a 1-D array over observations, averaged over chains/draws.
    """
    ll = model.trace.log_likelihood
    var = list(ll.data_vars)[0]
    pointwise = np.asarray(ll[var].values).reshape(-1, ll[var].shape[-1])
    # log-sum-exp mean over samples -> lppd (elpd contribution per obs)
    with np.errstate(divide="ignore"):
        return np.log(np.exp(pointwise).mean(axis=0))


def loo_elpd_i(model) -> np.ndarray:
    """Pointwise elpd from ``model.loo()`` (PSIS-corrected, includes the
    p_loo penalty). Falls back to the raw lppd if PSIS is unavailable."""
    try:
        loo = model.loo()
        elpd_i = np.asarray(loo.elpd_i)
        return np.asarray(elpd_i).reshape(-1)
    except Exception:
        return pointwise_elpd(model)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def elpd_se(elpd_i: np.ndarray) -> float:
    """Standard error of the total elpd from pointwise values."""
    elpd_i = np.asarray(elpd_i, dtype=float)
    n = elpd_i.size
    if n < 2:
        return 0.0
    return float(np.sqrt(n * np.var(elpd_i, ddof=1)))


def elpd_diff_se(elpd_i_a: np.ndarray, elpd_i_b: np.ndarray) -> float:
    """Standard error of the elpd difference between two models."""
    d = np.asarray(elpd_i_a, dtype=float) - np.asarray(elpd_i_b, dtype=float)
    n = d.size
    if n < 2:
        return 0.0
    return float(np.sqrt(n * np.var(d, ddof=1)))


def loo_table(pointwise: dict[str, np.ndarray]) -> pd.DataFrame:
    """Ranked PSIS-LOO comparison table from pointwise elpd arrays.

    Parameters
    ----------
    pointwise : dict[str, np.ndarray]
        Candidate name -> pointwise elpd over (concatenated) observations.

    Returns
    -------
    pd.DataFrame
        Columns ``rank``, ``elpd``, ``se``, ``elpd_diff``, ``dse``,
        ``weight`` (pseudo-BMA), sorted best-first.
    """
    names = list(pointwise)
    if not names:
        return pd.DataFrame()
    elpd = {n: float(np.sum(pointwise[n])) for n in names}
    se = {n: elpd_se(pointwise[n]) for n in names}
    best = max(elpd, key=elpd.get)
    rows = []
    for n in names:
        d = elpd[n] - elpd[best]
        rows.append(
            {
                "name": n,
                "elpd": elpd[n],
                "se": se[n],
                "elpd_diff": d,
                "dse": elpd_diff_se(pointwise[n], pointwise[best]),
            }
        )
    table = pd.DataFrame(rows)
    table["weight"] = pseudo_bma_weights(pointwise)
    table["rank"] = table["elpd"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("elpd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model-averaging weights (strategy C)
# ---------------------------------------------------------------------------


def pseudo_bma_weights(
    pointwise: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 42
) -> np.ndarray:
    """Pseudo-BMA weights with a Bayesian bootstrap (Vehtari et al. 2017).

    Returns the weight of each candidate (same order as ``pointwise`` keys).

    Parameters
    ----------
    pointwise : dict[str, np.ndarray]
        Candidate name -> pointwise elpd.
    n_boot : int
        Number of Bayesian-bootstrap resamples of the pointwise elpd.
    seed : int
        Seed for the bootstrap (weights are reproducible).
    """
    keys = list(pointwise)
    if not keys:
        return np.array([])
    stacked = np.stack([np.asarray(pointwise[k], dtype=float) for k in keys])
    n_obs = stacked.shape[1]
    rng = np.random.default_rng(seed)
    # Bayesian bootstrap: Dirichlet(1, ..., 1) resampling weights.
    dirichlet = rng.dirichlet(np.ones(n_obs), size=n_boot)  # (n_boot, n_obs)
    boot_elpd = stacked @ dirichlet.T  # (n_candidates, n_boot)
    boot_elpd -= boot_elpd.max(axis=0, keepdims=True)
    boot_w = np.exp(boot_elpd)
    boot_w /= boot_w.sum(axis=0, keepdims=True)
    return boot_w.mean(axis=1)


def stacking_weights(pointwise: dict[str, np.ndarray]) -> np.ndarray:
    """Log-score stacking weights via constrained optimisation.

    Solves for the weight vector maximising the log of the averaged
    predictive density, i.e. minimising ``-sum_i log(sum_k w_k p_k(y_i))``
    subject to ``w >= 0``, ``sum w = 1``, using the log pointwise elpd.

    This is the standard stacking solution (Yao et al. 2018) on the log
    predictive scale; ``pseudo_bma_weights`` is used as the default because
    it is more stable on short series, but stacking is reported too.

    Parameters
    ----------
    pointwise : dict[str, np.ndarray]
        Candidate name -> pointwise elpd.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    keys = list(pointwise)
    if not keys:
        return np.array([])
    logp = np.stack(
        [np.asarray(pointwise[k], dtype=float) for k in keys]
    ).T  # (n_obs, K)
    n_cand = len(keys)

    def neg_log_score(w):
        # Stacking objective: maximise sum_i log( sum_k w_k p_k(y_i) ), where
        # p_k(y_i) = exp(elpd_ik) is the predictive density.
        log_avg = logsumexp(logp + np.log(w + 1e-300), axis=1)
        return -float(log_avg.sum())

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_cand
    res = minimize(
        neg_log_score,
        x0=np.full(n_cand, 1.0 / n_cand),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        return pseudo_bma_weights(pointwise)
    return np.clip(res.x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CRPS (strategy B)
# ---------------------------------------------------------------------------


def crps_ensemble(samples: np.ndarray, y: float) -> float:
    """Empirical CRPS from an ensemble of predictive samples.

    Uses the fair estimator
        CRPS = E|X - y| - 0.5 * E|X - X'|
    with X, X' iid from the predictive ensemble (Gneiting & Raftery 2007).

    Parameters
    ----------
    samples : np.ndarray
        1-D array of predictive samples.
    y : float
        Observed value.

    Returns
    -------
    float
        The CRPS (lower is better).
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    n = samples.size
    if n == 0:
        return float("nan")
    a = float(np.abs(samples - y).mean())
    b = float(np.abs(samples[:, None] - samples[None, :]).sum()) / (2.0 * n * n)
    return a - b


def crps_ensemble_grid(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS for an (n_samples, n_points) ensemble vs a (n_points,) target."""
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.array([crps_ensemble(samples[:, j], y[j]) for j in range(y.size)])
