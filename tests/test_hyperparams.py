"""Tests for the Bayesian hyperparameter-selection workflow
(fixed_case_studies/hyperparams/).

Pure-logic tests (scoring, weights, recommendation rule, folds, candidates)
run fast; the few fitting tests use tiny data and minimal inference settings.
"""

import numpy as np
import pandas as pd
import pytest

from fixed_case_studies.hyperparams import bayesian, candidates, robustness, scoring, stacking, ts_cv


# ---------------------------------------------------------------------------
# scoring: CRPS
# ---------------------------------------------------------------------------


class TestCrps:
    def test_point_mass(self):
        """CRPS of a point mass at x evaluated at y is |x - y|."""
        samples = np.ones(50) * 3.0
        assert scoring.crps_ensemble(samples, 3.0) == pytest.approx(0.0, abs=1e-9)
        assert scoring.crps_ensemble(samples, 5.0) == pytest.approx(2.0, abs=1e-9)

    def test_symmetric_shift_invariance(self):
        """CRPS is translation-invariant in the mean shift."""
        a = scoring.crps_ensemble(np.random.default_rng(0).normal(0, 1, 500), 0.3)
        b = scoring.crps_ensemble(np.random.default_rng(0).normal(10, 1, 500), 10.3)
        assert a == pytest.approx(b, abs=0.05)

    def test_tighter_is_better(self):
        """A predictive ensemble concentrated on y scores better than a
        diffuse one."""
        rng = np.random.default_rng(1)
        y = 0.0
        tight = scoring.crps_ensemble(rng.normal(0, 0.1, 500), y)
        wide = scoring.crps_ensemble(rng.normal(0, 2.0, 500), y)
        assert tight < wide

    def test_grid(self):
        ens = np.random.default_rng(2).normal(0, 1, (200, 3))
        out = scoring.crps_ensemble_grid(ens, np.array([0.0, 1.0, -1.0]))
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# scoring: elpd tables and weights
# ---------------------------------------------------------------------------


class TestElpdTables:
    def test_loo_table_ranks_best_first(self):
        rng = np.random.default_rng(0)
        pointwise = {
            "a": rng.normal(-1.0, 1.0, 200),
            "b": rng.normal(-0.5, 1.0, 200),
            "c": rng.normal(0.0, 1.0, 200),
        }
        table = scoring.loo_table(pointwise)
        assert table.iloc[0]["name"] == "c"
        assert table.iloc[0]["elpd_diff"] == pytest.approx(0.0)
        assert np.all(table["dse"] >= 0)
        assert np.allclose(table["weight"].sum(), 1.0)

    def test_elpd_se_matches_manual_formula(self):
        elpd_i = np.arange(1.0, 21.0)
        n = len(elpd_i)
        manual = np.sqrt(n * np.var(elpd_i, ddof=1))
        assert scoring.elpd_se(elpd_i) == pytest.approx(manual)

    def test_elpd_diff_se(self):
        a = np.random.default_rng(3).normal(0, 1, 100)
        b = a + np.random.default_rng(4).normal(0, 0.5, 100)
        d = a - b
        manual = np.sqrt(len(d) * np.var(d, ddof=1))
        assert scoring.elpd_diff_se(a, b) == pytest.approx(manual)


class TestWeights:
    def test_pseudo_bma_weights_sum_to_one_and_favor_best(self):
        rng = np.random.default_rng(5)
        pointwise = {"good": rng.normal(0, 1, 300), "bad": rng.normal(-3, 1, 300)}
        w = scoring.pseudo_bma_weights(pointwise, n_boot=200, seed=0)
        assert np.allclose(w.sum(), 1.0)
        assert w[0] > 0.9  # 'good' dominates

    def test_stacking_weights(self):
        rng = np.random.default_rng(6)
        pointwise = {"a": rng.normal(0, 1, 200), "b": rng.normal(-2, 1, 200)}
        w = scoring.stacking_weights(pointwise)
        assert np.allclose(w.sum(), 1.0)
        assert w[0] > w[1]

    def test_weight_table(self):
        rng = np.random.default_rng(7)
        pointwise = {"x": rng.normal(0, 1, 100), "y": rng.normal(-1, 1, 100)}
        df = stacking.weight_table(pointwise, n_boot=100, seed=0)
        assert set(df["name"]) == {"x", "y"}
        assert np.allclose(df["pseudo_bma_weight"].sum(), 1.0)


# ---------------------------------------------------------------------------
# bayesian: recommendation rule
# ---------------------------------------------------------------------------


class TestRecommend:
    def _table(self, pairs, dse=1.0):
        """pairs: list of (candidate_value, elpd)."""
        names = [f"p={v}" for v, _ in pairs]
        elpds = [e for _, e in pairs]
        best = max(elpds)
        return pd.DataFrame(
            {
                "name": names,
                "elpd": elpds,
                "se": [1.0] * len(elpds),
                "elpd_diff": [e - best for e in elpds],
                "dse": [dse] * len(elpds),
                "weight": [0.25] * len(elpds),
            }
        )

    def test_keeps_frozen_when_within_dse(self):
        # best = 2.0 at p=0.5, frozen = 0.5 within 1 dse -> keep frozen
        table = self._table([(0.5, 2.0), (1.0, 1.5), (2.0, 1.0)])
        value, msg = bayesian.recommend(table, "p", current=0.5)
        assert value == 0.5
        assert "no evidence to change" in msg

    def test_prefers_smaller_within_dse(self):
        # best = 5.0; frozen = 3.0 is *outside* 1 dse, but 0.5 is inside
        # the indistinguishable set -> the smallest inside value wins.
        table = self._table([(0.5, 1.4), (1.0, 1.3), (3.0, 0.5), (5.0, 2.0)])
        value, msg = bayesian.recommend(table, "p", current=3.0)
        assert value == 0.5
        assert "smallest" in msg

    def test_clear_winner_when_far_apart(self):
        table = self._table([(0.5, -10.0), (1.0, -9.0), (5.0, 2.0)], dse=0.1)
        value, msg = bayesian.recommend(table, "p", current=0.5)
        assert value == 5.0
        assert "best candidate" in msg


# ---------------------------------------------------------------------------
# bayesian: prior-predictive sweep (tiny real fit)
# ---------------------------------------------------------------------------


class TestPriorPredictiveSweep:
    def test_sweep_on_tiny_data(self, sample_data):
        from vangja import FourierSeasonality, LinearTrend

        def factory(**kw):
            return LinearTrend(n_changepoints=0) + FourierSeasonality(7, 2, **kw)

        data = sample_data.copy()
        data["t"] = (data["ds"] - data["ds"].min()) / (data["ds"].max() - data["ds"].min())
        table = bayesian.prior_predictive_sweep(
            factory, data, "beta_sd", [0.1, 1.0],
            samples=100, seed=0, fit_kwargs={"method": "mapx"},
        )
        assert list(table.columns) == ["beta_sd", "coverage", "verdict"]
        assert len(table) == 2
        assert 0.0 <= table["coverage"].iloc[0] <= 1.0
        assert table["verdict"].iloc[0] in {"too wide", "target", "too tight"}


# ---------------------------------------------------------------------------
# ts_cv: folds and fold_future
# ---------------------------------------------------------------------------


class TestTsCv:
    def test_expanding_folds_structure(self):
        ds = pd.date_range("2020-01-01", periods=60, freq="D")
        train = pd.DataFrame({"ds": ds, "y": np.arange(60.0), "series": "s"})
        folds = ts_cv.expanding_folds(train, n_initial=30, step=7, horizon=7)
        assert len(folds) == 4  # 30..51 -> 30, 37, 44, 51 (51+7=58 <= 60)
        for tr, hold in folds:
            assert len(tr) >= 30
            assert len(hold) == 7
            assert tr["ds"].max() < hold["ds"].min()

    def test_fold_future_uses_model_scale(self):
        class FakeModel:
            t_scale_params = {"ds_min": pd.Timestamp("2020-01-01"),
                              "ds_max": pd.Timestamp("2020-02-29")}

        held = pd.DataFrame({"ds": pd.date_range("2020-02-10", periods=3, freq="D")})
        future = ts_cv.fold_future(FakeModel(), held)
        assert "t" in future.columns
        t0 = (held["ds"].iloc[0] - pd.Timestamp("2020-01-01")).days / 59.0
        assert future["t"].iloc[0] == pytest.approx(t0)


# ---------------------------------------------------------------------------
# robustness + candidates
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_perturb_continuous(self):
        assert robustness.perturb_values(1.0, continuous=True) == [0.5, 1.0, 1.5]

    def test_perturb_discrete_floor_at_1(self):
        assert robustness.perturb_values(2, continuous=False) == [1, 2, 3]
        assert robustness.perturb_values(1, continuous=False) == [1, 1, 2]


class TestCandidates:
    def test_current_in_candidates(self):
        candidates.assert_current_in_candidates("smart_home")
        candidates.assert_current_in_candidates("stocks")

    def test_baseline_included(self):
        for study in ("smart_home", "stocks"):
            grids, current = {
                "smart_home": (candidates.SMART_HOME, candidates.SMART_HOME_CURRENT),
                "stocks": (candidates.STOCKS, candidates.STOCKS_CURRENT),
            }[study]
            for param, values in grids.items():
                assert current[param] in values
