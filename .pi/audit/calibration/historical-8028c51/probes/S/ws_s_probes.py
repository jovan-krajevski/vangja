#!/usr/bin/env python
"""WS-S reproduction probes for the calibration snapshot main@8028c51.

Every probe re-derives a claim from the frozen code with a deterministic fixture.

Usage:
  PYTHONPATH=/home/jovan/repos/vangja/.pi/state/calibration/snapshot/src \
  /home/jovan/repos/vangja/.venv/bin/python ws_s_probes.py <probe>

Probes: lt_loss_signs fs_lambda delta_side_transfer partial_zero_cp
        partial_delta_tune_flag prior_from_idata_cov shift_for_tune mapx_predict_constants
"""
import copy
import sys
import warnings

import numpy as np
import pandas as pd
import pytensor
import xarray as xr

warnings.filterwarnings("ignore")

from vangja import (
    BetaConstant,
    FlatTrend,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)


class FakeIdata:
    """Minimal stand-in for an InferenceData/DataTree with a posterior group."""

    def __init__(self, posterior: xr.Dataset):
        self._p = posterior

    def __getitem__(self, key):
        if key == "posterior":
            return self._p
        raise KeyError(key)


def make_data(n=40, n_groups=1, y=3.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        ds = pd.date_range("2020-01-01", periods=n, freq="D")
        rows.append(
            pd.DataFrame(
                {"ds": ds, "y": y + rng.normal(0, 0.1, n), "series": f"s{g}"}
            )
        )
    return pd.concat(rows, ignore_index=True)


def slope_posterior(mean=3.0, sd=0.5, n_groups=None, n_cp=0, seed=0):
    rng = np.random.default_rng(seed)
    vars_ = {}
    shape = (1, 400) + ((n_groups,) if n_groups else ())
    dims = ("chain", "draw") + (("group",) if n_groups else ())
    vars_["lt_0 - slope"] = (dims, rng.normal(mean, sd, size=shape))
    if n_cp:
        vars_["lt_0 - delta"] = (("chain", "draw", "cp"), rng.normal(0, 0.05, (1, 400, n_cp)))
    return xr.Dataset(vars_)


def pot_value(pot, inputs, values):
    f = pytensor.function(inputs, pot)
    return float(np.asarray(f(*values)))


# ---------------------------------------------------------------------------
def probe_lt_loss_signs():
    """Evaluate the LinearTrend transfer loss potential at w=mu and w=mu+0.7."""
    data = make_data(n=40, n_groups=2)
    for pool in ("complete", "partial", "individual"):
        ng = 1 if pool == "complete" else 2
        idata = FakeIdata(slope_posterior(n_groups=(None if pool == "complete" else ng), n_cp=2))
        lt = LinearTrend(
            n_changepoints=2,
            pool_type=pool,
            tune_method="parametric",
            loss_factor_for_tune=1.0,
        )
        lt.fit(data, method="map", idata=idata, progressbar=False)
        mu, _ = lt._get_slope_params_from_idata(idata)
        pot = lt.model.named_vars["lt_0 - slope - loss"]
        if pool == "complete":
            slope_rv = lt.model.named_vars["lt_0 - slope"]
            v0 = pot_value(pot, [slope_rv], [np.array(float(mu))])
            v1 = pot_value(pot, [slope_rv], [np.array(float(mu) + 0.7)])
        elif pool == "individual":
            slope_rv = lt.model.named_vars["lt_0 - slope"]
            v0 = pot_value(pot, [slope_rv], [np.full(ng, float(mu))])
            v1 = pot_value(pot, [slope_rv], [np.full(ng, float(mu) + 0.7)])
        else:
            sh = lt.model.named_vars["lt_0 - slope_shared"]
            zo = lt.model.named_vars["lt_0 - slope_z_offset"]
            sg = lt.model.named_vars["lt_0 - slope_sigma"]
            v0 = pot_value(pot, [sh, zo, sg], [float(mu), np.zeros(ng), 1.0])
            v1 = pot_value(pot, [sh, zo, sg], [float(mu), np.full(ng, 0.7), 1.0])
        print(
            f"PROBE lt_loss_signs pool={pool:10s} slope_mu={float(mu):.4f} "
            f"potential(mu)={v0:+.6f} potential(mu+0.7)={v1:+.6f} "
            f"delta={v1 - v0:+.6f} direction={'REWARD' if v1 > v0 else 'PENALTY'}"
        )


# ---------------------------------------------------------------------------
def probe_fs_lambda():
    """Recover the Fourier loss scale lambda from the potential at beta=0."""
    period = 365.25
    n = 40
    data = make_data(n=n, n_groups=2)
    for pool in ("complete", "partial", "individual"):
        ng = 1 if pool == "complete" else 2
        rng = np.random.default_rng(0)
        beta_post = rng.normal(0.1, 0.05, (1, 500, 2))
        idata = FakeIdata(xr.Dataset({"fs_0 - beta(p=365.25,n=1)": (("chain", "draw", "b"), beta_post)}))
        fs = FourierSeasonality(
            period=period,
            series_order=1,
            pool_type=pool,
            tune_method="parametric",
            loss_factor_for_tune=1.0,
        )
        fs.fit(data, method="map", idata=idata, progressbar=False)
        beta_mean, _ = fs._get_beta_params_from_idata(idata)
        reg_ds = pd.DataFrame(
            {"ds": pd.date_range("2000-01-01", periods=int(np.ceil(period)), freq="D")}
        )
        reg_x = fs._fourier_series(reg_ds)
        old = reg_x @ beta_mean
        old_norm2 = float(old @ old)
        pot = fs.model.named_vars[
            "fs_0 - beta(p=365.25,n=1) - loss"
        ]
        # activate the one-sided penalty: new amplitude must EXCEED old
        k = 10.0
        if pool == "complete":
            beta_rv = fs.model.named_vars["fs_0 - beta(p=365.25,n=1)"]
            v0 = pot_value(pot, [beta_rv], [k * beta_mean])
        elif pool == "individual":
            beta_rv = fs.model.named_vars["fs_0 - beta(p=365.25,n=1)"]
            v0 = pot_value(pot, [beta_rv], [np.tile(k * beta_mean, (ng, 1))])
        else:
            sh = fs.model.named_vars["fs_0 - beta_shared"]
            zo = fs.model.named_vars["fs_0 - beta_z_offset(p=365.25,n=1)"]
            sg = fs.model.named_vars["fs_0 - beta_sigma(p=365.25,n=1)"]
            v0 = pot_value(pot, [sh, zo, sg], [k * beta_mean, np.zeros((ng, 2)), np.ones(2)])
        # potential = loss * lam * (||old||^2 - ||new||^2) = lam * (1 - k^2) * old_norm2
        lam_code = v0 / ((1.0 - k**2) * old_norm2)
        doc_formula = 2 * period / n
        print(
            f"PROBE fs_lambda pool={pool:10s} period={period} n_per_group={n} "
            f"||old||^2={old_norm2:.4f} potential(beta={k:.0f}*mean)={v0:+.4f} "
            f"lambda_code={lam_code:.4f} doc_lambda_2p_over_n={doc_formula:.4f}"
        )


# ---------------------------------------------------------------------------
def probe_delta_side_transfer():
    """Slope-moment extraction depends on the TARGET delta_side, not the source."""
    for label, slope_val, delta_vals in (
        ("left-style source  (slope=3, delta=+1,+1)", 3.0, np.array([1.0, 1.0])),
        ("right-style source (slope=5, delta=-1,-1)", 5.0, np.array([-1.0, -1.0])),
    ):
        delta_post = np.broadcast_to(delta_vals, (1, 400, 2)).copy()
        post = xr.Dataset(
            {
                "lt_0 - slope": (("chain", "draw"), np.full((1, 400), slope_val)),
                "lt_0 - delta": (("chain", "draw", "cp"), delta_post),
            }
        )
        idata = FakeIdata(post)
        out = []
        for side in ("left", "right"):
            lt = LinearTrend(delta_side=side, tune_method="parametric", n_changepoints=2)
            lt.model_idx = 0
            mean, sd = lt._get_slope_params_from_idata(idata)
            out.append(f"target delta_side={side:5s} -> prior slope mean={float(mean):+.3f}")
        print(f"PROBE delta_side_transfer | {label} | " + " | ".join(out))


# ---------------------------------------------------------------------------
def probe_partial_zero_cp():
    """LinearTrend(pool_type='partial', n_changepoints=0) under tuning."""
    data = make_data(n=40, n_groups=2)
    # non-tuned
    lt = LinearTrend(n_changepoints=0, pool_type="partial")
    lt.fit(data, method="map", progressbar=False)
    delta = lt.model.named_vars["lt_0 - delta"]
    print(
        "PROBE partial_zero_cp non-tuned: fit=OK, 'lt_0 - delta' in free_RVs="
        f"{delta in lt.model.free_RVs}, logp(delta)={lt.model.point_logps()['lt_0 - delta']:.3f}"
    )
    # parametric with delta present in the source posterior
    idata = FakeIdata(slope_posterior(n_groups=2, n_cp=2))
    for tune in ("parametric", "prior_from_idata"):
        m = LinearTrend(
            n_changepoints=0, pool_type="partial", tune_method=tune
        )
        try:
            m.fit(data, method="map", idata=idata, progressbar=False)
            print(f"PROBE partial_zero_cp tune={tune:17s}: fit=OK (unexpected)")
        except Exception as exc:  # noqa: BLE001
            print(
                f"PROBE partial_zero_cp tune={tune:17s}: fit=FAILED "
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
            )


# ---------------------------------------------------------------------------
def probe_partial_delta_tune_flag():
    """Partial pooling tunes deltas whenever tune_method='parametric'."""
    data = make_data(n=40, n_groups=2)
    # source posterior WITHOUT delta, target n_changepoints=2, delta_tune_method=None
    idata = FakeIdata(slope_posterior(n_groups=2, n_cp=0))
    for pool in ("complete", "partial", "individual"):
        m = LinearTrend(
            n_changepoints=2,
            pool_type=pool,
            tune_method="parametric",
            delta_tune_method=None,
        )
        try:
            m.fit(data, method="map", idata=idata, progressbar=False)
            print(
                f"PROBE partial_delta_tune_flag pool={pool:10s} tune_method=parametric "
                "delta_tune_method=None: fit=OK"
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"PROBE partial_delta_tune_flag pool={pool:10s} tune_method=parametric "
                f"delta_tune_method=None: fit=FAILED {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:110]}"
            )


# ---------------------------------------------------------------------------
def probe_prior_from_idata_cov():
    """Correlation of transferred Fourier beta under complete vs individual pooling."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (1, 500))
    corr = 0.9
    b0 = base
    b1 = corr * base + np.sqrt(1 - corr**2) * rng.normal(0, 1, (1, 500))
    beta_post = np.stack([b0, b1], axis=-1)
    idata = FakeIdata(
        xr.Dataset({"fs_0 - beta(p=7,n=1)": (("chain", "draw", "b"), beta_post)})
    )
    src_corr = float(np.corrcoef(beta_post[0, :, 0], beta_post[0, :, 1])[0, 1])
    data = make_data(n=20, n_groups=2, seed=4)
    for pool in ("complete", "individual"):
        fs = FourierSeasonality(
            period=7, series_order=1, pool_type=pool, tune_method="prior_from_idata"
        )
        fs.fit(data, method="map", idata=idata, progressbar=False)
        rng2 = np.random.default_rng(123)
        import pymc as pm

        with fs.model:
            pp = pm.sample_prior_predictive(draws=2000, random_seed=rng2)
        samples = pp.prior["fs_0 - beta(p=7,n=1)"].values.reshape(-1, 2)
        beta = fs.model.named_vars["fs_0 - beta(p=7,n=1)"]
        target_corr = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
        print(
            f"PROBE prior_from_idata_cov pool={pool:10s} source_corr={src_corr:.3f} "
            f"target_prior_corr={target_corr:.3f} beta_is_free_RV={beta in fs.model.free_RVs}"
        )


# ---------------------------------------------------------------------------
def probe_shift_for_tune():
    """shift_for_tune=True creates no 'shift' variable."""
    rng = np.random.default_rng(0)
    idata = FakeIdata(
        xr.Dataset(
            {"fs_0 - beta(p=7,n=1)": (("chain", "draw", "b"), rng.normal(0, 0.1, (1, 200, 2)))}
        )
    )
    data = make_data(n=30, n_groups=1, seed=5)
    fs = FourierSeasonality(
        period=7,
        series_order=1,
        tune_method="parametric",
        shift_for_tune=True,
    )
    fs.fit(data, method="map", idata=idata, progressbar=False)
    shift_vars = [k for k in fs.model.named_vars if "shift" in k]
    print(
        "PROBE shift_for_tune: shift_for_tune=True -> variables containing 'shift' in "
        f"named_vars: {shift_vars}"
    )


# ---------------------------------------------------------------------------
def probe_mapx_predict_constants():
    """predict() after mapx + prior_from_idata for constant/trend components."""
    data = make_data(n=30, n_groups=1, y=0.5, seed=3)
    rng = np.random.default_rng(0)
    fs_key = "fs_0 - beta(p=7,n=1)"
    fs_post = rng.normal(0, 0.1, (1, 200, 2))
    cases = {
        "FlatTrend": (
            FlatTrend(tune_method="prior_from_idata"),
            "ft_0 - intercept",
            rng.normal(5, 0.3, (1, 200)),
        ),
        "NormalConstant": (
            NormalConstant(mu=0, sd=1, tune_method="prior_from_idata"),
            "nc_0 - c(mu=0,sd=1)",
            rng.normal(0.5, 0.1, (1, 200)),
        ),
        "UniformConstant": (
            UniformConstant(-1, 1, tune_method="prior_from_idata"),
            "uc_0 - c(l=-1,u=1)",
            rng.uniform(-0.5, 0.5, (1, 200)),
        ),
        "BetaConstant": (
            BetaConstant(0, 1, tune_method="prior_from_idata"),
            "bc_0 - c(l=0,u=1)",
            rng.uniform(0.2, 0.8, (1, 200)),
        ),
    }
    for name, (comp, key, vals) in cases.items():
        idata = FakeIdata(
            xr.Dataset(
                {
                    key: (("chain", "draw"), vals),
                    fs_key: (("chain", "draw", "b"), fs_post),
                }
            )
        )
        model = comp + FourierSeasonality(7, 1, tune_method="prior_from_idata")
        try:
            model.fit(data, method="mapx", idata=idata, progressbar=False)
            try:
                model.predict(horizon=3)
                print(f"PROBE mapx_predict_constants {name}: mapx+predict=OK")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"PROBE mapx_predict_constants {name}: mapx fit OK, predict "
                    f"FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"PROBE mapx_predict_constants {name}: mapx fit FAILED "
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
            )


# ---------------------------------------------------------------------------
def probe_bc_individual_ignores_priors():
    """BetaConstant individual + prior_from_idata never consumes the prior."""
    from pytensor.graph.basic import ancestors

    rng = np.random.default_rng(0)
    idata = FakeIdata(
        xr.Dataset(
            {"bc_0 - c(l=0,u=1)": (("chain", "draw"), rng.uniform(0.2, 0.8, (1, 200)))}
        )
    )
    data = make_data(n=30, n_groups=2, y=0.5, seed=6)
    bc = BetaConstant(0, 1, pool_type="individual", tune_method="prior_from_idata")
    bc.fit(data, method="map", idata=idata, progressbar=False)
    beta_rv = bc.model.named_vars["bc_0 - beta(a=0.5,b=0.5)"]
    prior_det = bc.model.named_vars["prior_bc_0 - c(l=0,u=1)"]
    graph = ancestors([bc.model["obs"]])
    used = any(node is prior_det for node in graph)
    print(
        "PROBE bc_individual_ignores_priors: "
        f"'prior_bc_0 - c(l=0,u=1)' created={prior_det is not None}, "
        f"is ancestor of obs={used}, beta RV op={type(beta_rv.owner.op).__name__}"
    )


PROBES = {
    "lt_loss_signs": probe_lt_loss_signs,
    "fs_lambda": probe_fs_lambda,
    "delta_side_transfer": probe_delta_side_transfer,
    "partial_zero_cp": probe_partial_zero_cp,
    "partial_delta_tune_flag": probe_partial_delta_tune_flag,
    "prior_from_idata_cov": probe_prior_from_idata_cov,
    "shift_for_tune": probe_shift_for_tune,
    "mapx_predict_constants": probe_mapx_predict_constants,
    "bc_individual_ignores_priors": probe_bc_individual_ignores_priors,
}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    if name == "all":
        for probe in PROBES.values():
            probe()
    else:
        PROBES[name]()
