# Chapter 11: Uncertainty Estimation in Vangja

## Overview

Vangja provides prediction intervals via the `predict_uncertainty()` method. The approach differs depending on the inference method used:

- **MCMC / VI methods** (NUTS, Metropolis, ADVI, etc.): Posterior draw propagation — each posterior sample is pushed through the model to produce a family of trajectories, yielding natural Bayesian credible intervals.
- **MAP methods** (MAP, MAPX): Residual-calibrated intervals — a hybrid approach combining the fitted observation noise, in-sample residual calibration, and forecast-distance scaling.

Both approaches are described in detail below.

---

## 1. MCMC / VI: Posterior Draw Propagation

### Idea

When the model is fitted with MCMC or VI, the `trace` object contains samples from the posterior distribution over all model parameters. Each posterior draw defines a complete set of parameter values; pushing a draw through the forward model produces one possible prediction trajectory. The spread of these trajectories quantifies parameter uncertainty.

### Algorithm

Given `N` posterior draws (across chains and draws):

1. **Sub-sample** `S = min(uncertainty_samples, N)` draws uniformly at random (seed 42 for reproducibility).
2. For each sampled draw index `(chain, draw)`:
   - Extract all parameter values into a dictionary `θ_s = {var: posterior[var][chain, draw]}`.
   - Create a fresh copy of the future DataFrame (to avoid side effects from component column writes).
   - Call `self._predict(future_copy, "mapx", θ_s, None)` — this reuses the existing `_predict_map` code path, treating each posterior draw as if it were a MAP point estimate.
   - Collect the resulting prediction array of shape `(n_groups, T)`.
3. **Stack** all `S` prediction arrays into shape `(S, n_groups, T)`.
4. For each group, compute pointwise quantiles:
   - `yhat_lower = quantile(α/2)`
   - `yhat_upper = quantile(1 − α/2)`
   where `α = 1 − interval_width`.
5. The point forecast (`yhat`) is computed separately using the standard `_predict()` (which averages over the posterior for MCMC).
6. **Rescale** all values from the internal `[0, 1]` scale back to the original data scale using `y_min` and `y_max`.

### Why Reuse `_predict_map`?

The key design insight is that `_predict_map` in each component is a pure forward function: given parameter values in a dictionary, it computes predictions. By extracting individual posterior draws into the same dictionary format that MAP uses, we can reuse the entire prediction pipeline without modifying any component. This avoids:

- Adding per-draw prediction methods to every component.
- Modifying `_predict_mcmc` (which by design averages over draws).
- Any changes to the composition operators (`+`, `*`, `**`).

### Strengths

- **Exact**: Captures the full posterior uncertainty, including parameter correlations.
- **Component-agnostic**: Works for any model composition without component-level changes.
- **Standard Bayesian approach**: Produces proper credible intervals.

### Limitations

- Computationally expensive for large `uncertainty_samples` (each draw requires a full forward pass).
- Credible intervals are not calibrated in the frequentist sense (they represent posterior belief, not coverage guarantees).

---

## 2. MAP / MAPX: Residual-Calibrated Intervals

### Motivation

MAP and MAPX produce a single point estimate — there are no posterior draws to propagate. We need a different strategy to estimate prediction intervals.

The standard approach (used by Facebook Prophet) simulates future trend uncertainty by generating synthetic changepoints with Laplace-distributed magnitudes at Poisson-distributed times, then adds Gaussian observation noise estimated from in-sample residuals. However, Prophet's approach has notable limitations:

1. **Only trend has uncertainty** — seasonality and other components contribute zero uncertainty under MAP.
2. **Relies on changepoint extrapolation** — the synthetic future changepoints are a heuristic, not derived from the posterior.
3. **No epistemic uncertainty** — the intervals don't grow with forecast distance beyond what changepoints provide.

### Vangja's Approach

Vangja uses a **hybrid residual-calibrated** approach that addresses these limitations:

$$\hat{y}(t) \pm z_{\alpha/2,\nu} \cdot \hat{\sigma} \cdot \sqrt{1 + h / n}$$

where:

- $\hat{y}(t)$ is the MAP point prediction
- $z_{\alpha/2,\nu}$ is the Student-t quantile with $\nu = \max(n - 2, 1)$ degrees of freedom
- $\hat{\sigma}$ is the calibrated noise estimate (see below)
- $h$ is the forecast distance (days from end of training data, clamped to ≥ 0)
- $n$ is the number of training observations

#### Noise Estimation: $\hat{\sigma}$

The noise estimate combines two sources:

$$\hat{\sigma} = \max(\sigma_{\text{fitted}}, \sigma_{\text{residual}})$$

1. **Fitted sigma** ($\sigma_{\text{fitted}}$): The observation noise parameter from the MAP solution. In the PyMC model, `sigma` is a parameter of the likelihood that gets optimized jointly with all other parameters during MAP inference.

2. **Residual standard deviation** ($\sigma_{\text{residual}}$): The standard deviation of in-sample residuals $e_i = y_i - \hat{y}(t_i)$, computed per group. This captures any systematic underfitting not reflected in the fitted sigma.

Taking the max ensures the intervals are never narrower than what either estimate alone would suggest.

#### Forecast-Distance Scaling: $\sqrt{1 + h/n}$

This factor captures the intuition that predictions further from the training data should be more uncertain. At `h = 0` (start of forecast horizon, still within training range), the factor is 1.0. As `h` grows relative to `n`, the factor increases sub-linearly:

| Scenario | h/n | Scaling factor |
|----------|-----|---------------|
| Beginning of forecast | 0 | 1.00 |
| 10% of training length ahead | 0.1 | 1.05 |
| Half training length ahead | 0.5 | 1.22 |
| Full training length ahead | 1.0 | 1.41 |
| 2× training length ahead | 2.0 | 1.73 |

This is analogous to the prediction interval formula from linear regression, where the variance of a new prediction grows with distance from the center of the data.

#### Student-t Quantiles

We use Student-t quantiles instead of normal quantiles to account for the finite-sample uncertainty in estimating $\sigma$. With $\nu = n - 2$ degrees of freedom (for slope + intercept), this gives wider intervals for small training sets and converges to the normal distribution for large $n$.

### Per-Group Computation

All quantities ($\sigma_{\text{fitted}}$, $\sigma_{\text{residual}}$, $n$) are computed per-group when `pool_type` results in multiple series. Each series gets its own uncertainty intervals calibrated to its own noise characteristics.

### Advantages Over Prophet's Approach

| Aspect | Prophet (MAP) | Vangja (MAP) |
|--------|--------------|--------------|
| Trend uncertainty | Simulated future changepoints | Captured via $\sqrt{1+h/n}$ scaling |
| Seasonality uncertainty | None | Included in residual calibration |
| Component coupling | Independent (trend only) | Joint (via residuals of full model) |
| Epistemic growth | Changepoint-driven only | Explicit forecast-distance scaling |
| Small-sample correction | None (uses Normal) | Student-t with $n-2$ d.f. |
| Calibration | Heuristic | Residual-calibrated |

The key insight is that Prophet decomposes uncertainty by component (and only models trend uncertainty), while Vangja estimates total prediction uncertainty from the full model's behavior. This naturally captures interactions between components and doesn't rely on extrapolating changepoint patterns into the future.

### Limitations

- **Homoscedastic assumption**: The noise estimate $\hat{\sigma}$ is constant over time. If the data has heteroscedastic noise (e.g., variance that changes seasonally), the intervals may be too wide in quiet periods and too narrow in volatile periods.
- **Linear scaling heuristic**: The $\sqrt{1 + h/n}$ factor is borrowed from linear regression theory and may not perfectly capture nonlinear forecast degradation.
- **No component decomposition**: Unlike Prophet, we do not separately attribute uncertainty to trend vs. seasonality. The intervals are for the total prediction only.

---

## Usage

```python
from vangja.components import LinearTrend, FourierSeasonality

model = LinearTrend() + FourierSeasonality(365.25, 10)
model.fit(data, method="nuts", samples=1000, chains=4)

# Predict with 95% uncertainty intervals
future = model.predict_uncertainty(horizon=90, interval_width=0.95)

# The result contains yhat_<group>, yhat_lower_<group>, yhat_upper_<group>
print(future[["ds", "yhat_0", "yhat_lower_0", "yhat_upper_0"]].head())

# Plot with built-in support (uncertainty bands shown automatically)
model.plot(future)
```

For MAP inference:

```python
model = LinearTrend() + FourierSeasonality(365.25, 10)
model.fit(data, method="mapx")

# Works the same way — uses residual-calibrated intervals instead
future = model.predict_uncertainty(horizon=90)
model.plot(future)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | (required) | Number of future steps to forecast |
| `freq` | `"D"` | Frequency of forecast steps |
| `uncertainty_samples` | 200 | Number of posterior draws for MCMC/VI. Ignored for MAP. |
| `interval_width` | 0.95 | Width of prediction interval (e.g. 0.95 for 95%) |

---

## Implementation Notes

### Composable Models

The uncertainty estimation works transparently with composed models (`+`, `*`, `**`). Since `predict_uncertainty` operates at the top-level `TimeSeriesModel` (calling `_predict` which dispatches to the composed tree), no component-level changes are needed in either approach.

### Transfer Learning

When a model is fitted with transfer learning (`idata` parameter), the posterior or MAP estimate already incorporates the prior information from the base model. The uncertainty estimation works identically — for MCMC, the posterior draws reflect the transfer-informed distribution; for MAP, the residuals reflect the transfer-informed fit.

### Multi-Series

For multi-series models, uncertainty is computed per-group. Each group's intervals use that group's own residual statistics and noise parameters. The intervals are independent across groups (no cross-group uncertainty correlation is modeled).
