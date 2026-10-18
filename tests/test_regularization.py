"""Tests for transfer-learning regularization and transfer dispatch.

Covers the review findings:

- P0-2: trend regularisation potential must be a *negative quadratic* in all
  pooling modes (manuscript: ``-phi * (w - w_MAP)^2``).
- P0-3: seasonal regularisation scale must be the same per-grid-point
  normalization in all pooling modes, independent of sample size.
- P0-4: changepoint-transfer dispatch must read ``delta_tune_method`` (not
  ``tune_method``) in every pooling mode.
- P0-5: the transferred slope quantity is the end-of-history slope of the
  source model (``slope + sum(delta)`` under ``delta_side="left"``; the raw
  ``slope`` under ``delta_side="right"``).

All model-graph tests compile the Potential expression directly with
``pytensor.function`` so no sampling is required; they are fast and
deterministic.
"""

import math

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from vangja.components.fourier_seasonality import FourierSeasonality
from vangja.components.linear_trend import LinearTrend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_idata(posterior: dict[str, np.ndarray]) -> az.InferenceData:
    """Build a synthetic InferenceData from a nested dict (arviz >= 1.x)."""
    return az.from_dict({"posterior": posterior})


def _slope_idata(mu: float = 0.5, n_draws: int = 10) -> az.InferenceData:
    """Synthetic posterior containing only ``lt_0 - slope`` with a known mean."""
    return _make_idata({"lt_0 - slope": np.full((1, n_draws), mu)})


def _single_data(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=n),
            "y": np.zeros(n),
            "t": np.linspace(0, 1, n),
            "series": "s",
        }
    )


def _multi_data(n_per_series: int = 20, n_series: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_per_series)
    frames = []
    for i in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "ds": dates,
                    "y": np.zeros(n_per_series),
                    "t": np.linspace(0, 1, n_per_series),
                    "series": f"s{i}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _compile_potential(component, data, model, loss_name: str, param_name: str):
    """Compile the named Potential as a pytensor function of the parameter RV."""
    component.definition(model, data, {}, priors=None, idata=_slope_idata())
    param = model.named_vars[param_name]
    loss = model.named_vars[loss_name]
    return pytensor.function([param], loss)


# ---------------------------------------------------------------------------
# P0-2: trend regularisation potential — sign and form
# ---------------------------------------------------------------------------


class TestTrendRegularization:
    """The trend potential must be ``-loss * sum((slope - mu)^2)`` everywhere."""

    @pytest.mark.parametrize("pool_type", ["complete", "partial", "individual"])
    def test_zero_penalty_at_equality(self, pool_type):
        data = _single_data() if pool_type == "complete" else _multi_data()
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
            pool_type=pool_type,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, data, model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            if pool_type == "complete":
                value = f(0.5)
            else:
                value = f(np.full(lt.n_groups, 0.5))
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_complete_negative_quadratic_sign_and_form(self):
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _single_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            at_mu = f(0.5)
            at_d1 = f(1.5)
            at_d2 = f(2.5)
            at_md1 = f(-0.5)

        # Negative quadratic: -(1)^2 = -1, -(2)^2 = -4
        assert at_mu == pytest.approx(0.0, abs=1e-6)
        assert at_d1 == pytest.approx(-1.0)
        assert at_d2 == pytest.approx(-4.0)
        # Symmetric around the transferred mean
        assert at_md1 == pytest.approx(at_d1)

    def test_complete_increasingly_negative_with_distance(self):
        """Penalty must decrease (become more negative) as the slope departs."""
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _single_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            values = [f(0.5 + d) for d in (0.0, 1.0, 2.0, 3.0)]
        assert values[0] == pytest.approx(0.0, abs=1e-6)
        for closer, farther in zip(values[:-1], values[1:]):
            assert farther < closer

    def test_complete_correct_gradient_direction(self):
        """A regulariser must *penalise* deviation, not reward it."""
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _single_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
        # Moving away from mu must decrease the potential (negative gradient)
        assert f(1.5) < f(0.5)
        assert f(-0.5) < f(0.5)

    def test_individual_sums_groups(self):
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
            pool_type="individual",
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _multi_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            value = f(np.array([1.5, 2.5]))  # deviations 1 and 2
        assert value == pytest.approx(-(1.0 + 4.0))

    def test_partial_negative_quadratic(self):
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            n_changepoints=0,
            pool_type="partial",
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _multi_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            at_mu = f(np.array([0.5, 0.5]))
            at_d = f(np.array([1.5, 2.5]))
        assert at_mu == pytest.approx(0.0, abs=1e-6)
        assert at_d == pytest.approx(-5.0)

    def test_loss_factor_scales_penalty(self):
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=3.0,
            n_changepoints=0,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _single_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
            value = f(1.5)
        assert value == pytest.approx(-3.0)

    def test_zero_loss_factor_disables_potential(self):
        lt = LinearTrend(
            tune_method="parametric",
            loss_factor_for_tune=0.0,
            n_changepoints=0,
        )
        with pm.Model() as model:
            f = _compile_potential(
                lt, _single_data(), model, "lt_0 - slope - loss", "lt_0 - slope"
            )
        assert f(2.5) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# P0-3: seasonal regularisation scale
# ---------------------------------------------------------------------------


class TestSeasonalRegularizationScale:
    """The seasonal potential must use the same per-grid-point scale in all modes."""

    PERIOD = 60
    ORDER = 2
    BETA_MEAN = np.array([0.3, -0.2, 0.1, 0.4])

    @pytest.fixture()
    def fs_idata(self):
        beta_key = "fs_0 - beta"
        samples = np.tile(self.BETA_MEAN, (1, 10, 1))
        return _make_idata({beta_key: samples})

    def _expected_penalty(self, component, new_beta, n_groups=1, loss_factor=1.0):
        """Expected value: loss * (1/n_t) * sum_g min(0, ||old||^2 - ||new_g||^2)."""
        reg_ds = pd.DataFrame(
            {
                "ds": pd.date_range(
                    "2000-01-01", periods=math.ceil(self.PERIOD), freq="D"
                )
            }
        )
        reg_x = component._fourier_series(reg_ds)
        n_t = len(reg_ds)
        old = reg_x @ self.BETA_MEAN
        old_energy = float(old @ old)
        new = reg_x @ np.asarray(new_beta)
        new_energy = float(np.sum(new * new))
        return loss_factor * (1.0 / n_t) * min(0.0, old_energy - new_energy) * n_groups

    def _compile(self, pool_type, data):
        fs = FourierSeasonality(
            period=self.PERIOD,
            series_order=self.ORDER,
            tune_method="parametric",
            loss_factor_for_tune=1.0,
            pool_type=pool_type,
        )
        beta_key = "fs_0 - beta"
        model = pm.Model()
        with model:
            fs.definition(
                model,
                data,
                {},
                priors=None,
                idata=_make_idata({beta_key: np.tile(self.BETA_MEAN, (1, 10, 1))}),
            )
            beta = model.named_vars[beta_key]
            loss = model.named_vars[f"{beta_key} - loss"]
            f = pytensor.function([beta], loss)
        return fs, f

    @pytest.mark.parametrize("pool_type", ["complete", "partial", "individual"])
    def test_exact_scale(self, pool_type):
        data = _single_data() if pool_type == "complete" else _multi_data()
        fs, f = self._compile(pool_type, data)
        new_beta = 2.0 * self.BETA_MEAN  # 4x the energy -> penalty active
        if pool_type == "complete":
            value = f(new_beta)
            expected = self._expected_penalty(fs, new_beta, n_groups=1)
        else:
            value = f(np.stack([new_beta] * 2))
            expected = self._expected_penalty(fs, new_beta, n_groups=2)
        assert value == pytest.approx(expected)

    @pytest.mark.parametrize("pool_type", ["complete", "partial", "individual"])
    def test_zero_penalty_when_new_energy_not_larger(self, pool_type):
        data = _single_data() if pool_type == "complete" else _multi_data()
        _, f = self._compile(pool_type, data)
        smaller = 0.5 * self.BETA_MEAN  # lower energy -> penalty clamped at 0
        if pool_type == "complete":
            value = f(smaller)
        else:
            value = f(np.stack([smaller] * 2))
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_consistency_across_pooling_modes(self):
        """Equivalent grouping and sample size -> identical penalty in all modes."""
        new_beta = 2.0 * self.BETA_MEAN
        values = {}
        for pool_type in ["complete", "partial", "individual"]:
            data = (
                _single_data() if pool_type == "complete" else _multi_data(n_series=1)
            )
            _, f = self._compile(pool_type, data)
            values[pool_type] = (
                f(new_beta) if pool_type == "complete" else f(new_beta[None, :])
            )
        assert values["partial"] == pytest.approx(values["complete"])
        assert values["individual"] == pytest.approx(values["complete"])

    def test_gate_inactive_when_history_covers_half_period(self):
        """No penalty when every series has at least half a period of data."""
        fs, f = self._compile("complete", _single_data(n=40))  # 60 <= 2*40
        new_beta = 2.0 * self.BETA_MEAN
        assert f(new_beta) == pytest.approx(0.0, abs=1e-6)

    def test_scale_independent_of_sample_size(self):
        """Same penalty for different training sizes (both below the gate)."""
        fs_small, f_small = self._compile("complete", _single_data(n=10))
        fs_large, f_large = self._compile("complete", _single_data(n=25))
        new_beta = 2.0 * self.BETA_MEAN
        assert f_small(new_beta) == pytest.approx(f_large(new_beta))
        assert f_small(new_beta) == pytest.approx(
            self._expected_penalty(fs_small, new_beta)
        )


# ---------------------------------------------------------------------------
# P0-4: changepoint-transfer dispatch
# ---------------------------------------------------------------------------


class TestChangepointTransferDispatch:
    """Changepoint transfer must be governed by ``delta_tune_method`` only."""

    def _make_lt_idata(self):
        return _make_idata(
            {
                "lt_0 - slope": np.full((1, 10), 0.5),
                "lt_0 - delta": np.full((1, 10, 3), 0.1),
            }
        )

    def _priors(self):
        return {
            "prior_lt_0 - slope": pt.as_tensor(0.5),
            "prior_lt_0 - delta": pt.as_tensor(np.zeros(3)),
        }

    def _build_delta(self, pool_type, delta_tune_method):
        idata = self._make_lt_idata()
        priors = self._priors()
        data = _single_data() if pool_type == "complete" else _multi_data()
        lt = LinearTrend(
            tune_method="prior_from_idata",
            delta_tune_method=delta_tune_method,
            n_changepoints=3,
            pool_type=pool_type,
        )
        with pm.Model() as model:
            lt.definition(model, data, {}, priors=priors, idata=idata)
            delta = model.named_vars["lt_0 - delta"]
        return model, delta

    @pytest.mark.parametrize(
        "pool_type",
        ["complete", "partial", "individual"],
    )
    def test_none_disables_transfer(self, pool_type):
        """delta_tune_method=None -> no changepoint transfer, free Laplace RV."""
        model, delta = self._build_delta(pool_type, None)
        assert delta in model.free_RVs

    @pytest.mark.parametrize("pool_type", ["complete", "partial"])
    def test_prior_from_idata_transfers(self, pool_type):
        """delta_tune_method='prior_from_idata' -> delta fixed to the prior."""
        model, delta = self._build_delta(pool_type, "prior_from_idata")
        assert delta not in model.free_RVs

    def test_individual_prior_from_idata_free_laplace(self):
        """Individual pooling: free per-group Laplace centered at the prior."""
        model, delta = self._build_delta("individual", "prior_from_idata")
        assert delta in model.free_RVs
        assert isinstance(delta.owner.op, pm.Laplace)

    @pytest.mark.parametrize(
        "pool_type",
        ["complete", "partial", "individual"],
    )
    def test_parametric_transfers_params(self, pool_type):
        """delta_tune_method='parametric' -> free Laplace with transferred params."""
        model, delta = self._build_delta(pool_type, "parametric")
        assert delta in model.free_RVs
        assert isinstance(delta.owner.op, pm.Laplace)

    @pytest.mark.parametrize(
        "pool_type",
        ["complete", "partial", "individual"],
    )
    def test_tune_method_differs_from_delta_tune_method(self, pool_type):
        """tune_method='prior_from_idata' but delta_tune_method=None -> no
        changepoint transfer, regardless of what the slope does."""
        lt = LinearTrend(
            tune_method="prior_from_idata",
            delta_tune_method=None,
            n_changepoints=3,
            pool_type=pool_type,
        )
        idxs: dict[str, int] = {}
        lt._assign_model_idx(idxs)
        names = lt._get_prior_var_names()
        assert names == ["lt_0 - slope"]


# ---------------------------------------------------------------------------
# P0-5: slope-transfer definition
# ---------------------------------------------------------------------------


class TestSlopeTransferDefinition:
    """The transferred quantity is the end-of-history slope of the source."""

    SLOPE = np.array([[0.1, 0.2, 0.3, 0.4]])  # (1 chain, 4 draws)
    DELTA = np.array(
        [
            [
                [0.05, 0.1, -0.2],
                [0.0, 0.1, 0.1],
                [-0.1, 0.2, 0.3],
                [0.2, -0.05, 0.0],
            ]
        ]
    )  # (1, 4, 3)

    def _make_idata_with_delta(self):
        return _make_idata(
            {
                "lt_0 - slope": self.SLOPE,
                "lt_0 - delta": self.DELTA,
            }
        )

    def test_left_side_transfers_end_of_history_slope(self):
        """delta_side='left': end slope = slope + sum(delta) (all changepoints
        precede the end of history)."""
        lt = LinearTrend(n_changepoints=3, delta_side="left")
        lt.model_idx = 0
        slope_mean, slope_sd = lt._get_slope_params_from_idata(
            self._make_idata_with_delta()
        )
        end_slopes = self.SLOPE + self.DELTA.sum(axis=-1)
        assert slope_mean == pytest.approx(end_slopes.mean())
        assert slope_sd == pytest.approx(end_slopes.std())

    def test_right_side_transfers_base_slope(self):
        """delta_side='right': the slope parameter already is the end slope."""
        lt = LinearTrend(n_changepoints=3, delta_side="right")
        lt.model_idx = 0
        slope_mean, slope_sd = lt._get_slope_params_from_idata(
            self._make_idata_with_delta()
        )
        assert slope_mean == pytest.approx(self.SLOPE.mean())
        assert slope_sd == pytest.approx(self.SLOPE.std())

    def test_missing_delta_posterior_leaves_slope_unchanged(self):
        """If the posterior has no delta variable, the raw slope is used."""
        lt = LinearTrend(n_changepoints=3, delta_side="left")
        lt.model_idx = 0
        slope_mean, slope_sd = lt._get_slope_params_from_idata(_slope_idata(mu=0.5))
        assert slope_mean == pytest.approx(0.5)
        assert slope_sd == pytest.approx(0.0, abs=1e-6)

    def test_override_mean_skips_adjustment(self):
        lt = LinearTrend(
            n_changepoints=3,
            delta_side="left",
            override_slope_mean_for_tune=np.array(7.0),
        )
        lt.model_idx = 0
        slope_mean, _ = lt._get_slope_params_from_idata(self._make_idata_with_delta())
        assert slope_mean == pytest.approx(7.0)
