"""Bayesian hyperparameter selection for the smart-home case study.

    python fixed_case_studies/smart_home/02_select_hyperparams.py [--no-verify]

Selects the Prophet-inherited hyperparameters
(``HYPERPARAMETER_SELECTION.md``):

- ``FlatTrend.intercept_sd`` (frozen 0.1)
- ``FourierSeasonality.beta_sd`` (frozen 0.25)
- ``FourierSeasonality.series_order`` yearly (frozen 5) and weekly (frozen 2)

The workflow runs entirely on **training windows** (the 91-day training
blocks of the ``origin_apr`` / ``origin_jul`` splits); the 260-day test
horizon is never used.  Evidence: prior-predictive calibration (sd's),
coordinate-wise PSIS-LOO (ADVI screening + optional small-NUTS
verification of the top-2), leave-future-out CRPS, model-averaging weights,
a robustness sweep, and a full-Bayes hyperprior demonstration.

Outputs (gitignored): ``results_hyperparams/`` — per-hyperparameter reports
+ ``smart_home_recommendations.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from vangja import FlatTrend, FourierSeasonality, UniformConstant

from fixed_case_studies.hyperparams import bayesian, candidates, full_bayes, robustness, stacking, ts_cv

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "results_hyperparams"
SMART_HOME_COLUMNS = [
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
]
TEMP_COLUMN = "Boston"
DATA_END = "2016-12-16"


# ---------------------------------------------------------------------------
# Frozen data (identical to the study, read from the local copies)
# ---------------------------------------------------------------------------


def load_frozen_splits(train_end: str):
    """Read the frozen local CSVs and cut at ``train_end`` (no future leak)."""
    sh = pd.read_csv(DATA_DIR / "smart_home_daily.csv", parse_dates=["ds"])
    train = sh[sh["ds"] < pd.Timestamp(train_end)].copy()
    tag = train_end.replace("-", "")
    temp = pd.read_csv(DATA_DIR / f"boston_temp_until_{tag}.csv", parse_dates=["ds"])
    return train, temp


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def build_target(
    *,
    intercept_sd: float = 0.1,
    beta_sd: float = 0.25,
    yearly_order: int = 5,
    weekly_order: int = 2,
    tune_method=None,
    pool: str = "partial",
    shrinkage: int = 1,
):
    """Mirror of ``runner.build_target_model`` with the hyperparameters
    exposed as keyword arguments (the study's main configuration)."""
    trend = FlatTrend(
        intercept_mean=0.5, intercept_sd=intercept_sd, pool_type="individual"
    )
    yearly = FourierSeasonality(
        period=365.25,
        series_order=yearly_order,
        beta_sd=beta_sd,
        pool_type=pool,
        tune_method=tune_method,
        shrinkage_strength=shrinkage,
    )
    weekly = FourierSeasonality(
        period=7,
        series_order=weekly_order,
        beta_sd=beta_sd,
        pool_type=pool,
        shrinkage_strength=shrinkage,
    )
    constant = UniformConstant(lower=-1, upper=1, pool_type=pool, shrinkage_strength=shrinkage)
    return trend + constant * yearly + weekly


def make_factory(**defaults):
    """Closure so the selection engine can vary one hyperparameter at a time."""
    def factory(**kw):
        return build_target(**{**defaults, **kw})
    return factory


# ---------------------------------------------------------------------------
# Temperature context 
# (ADVI for screening; the study's own NUTS source is used at run time)
# ---------------------------------------------------------------------------


def fit_temp_advi(temp_train: pd.DataFrame, seed: int, progressbar: bool = False):
    from fixed_case_studies.smart_home.runner import build_temp_model

    model = build_temp_model()
    model.fit(
        temp_train,
        scaler="minmax",
        method="advi",
        n=20000,
        samples=2000,
        random_seed=seed,
        progressbar=progressbar,
    )
    return model


def load_or_fit_temp_advi(temp_train: pd.DataFrame, split_label: str, seed: int):
    """Cached ADVI temperature fit, keyed by split x seed (like the runner's
    ``load_or_fit_temp_model`` but ADVI for screening speed)."""
    import json
    import pickle

    from fixed_case_studies.smart_home.runner import build_temp_model

    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"temp_advi__{split_label}__seed{seed}"
    pkl = cache_dir / f"{tag}.pkl"
    meta = cache_dir / f"{tag}.json"
    if pkl.exists() and meta.exists():
        model = build_temp_model()
        with open(pkl, "rb") as fh:
            model.trace = pickle.load(fh)
        t_scale = json.loads(meta.read_text())
        model.t_scale_params = {
            k: pd.Timestamp(v) if k in ("ds_min", "ds_max") else v
            for k, v in t_scale.items()
        }
        return model
    model = fit_temp_advi(temp_train, seed, progressbar=False)
    with open(pkl, "wb") as fh:
        pickle.dump(model.trace, fh)
    meta.write_text(
        json.dumps(
            {
                "ds_min": model.t_scale_params["ds_min"].isoformat(),
                "ds_max": model.t_scale_params["ds_max"].isoformat(),
            }
        )
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", nargs="+", default=["2016-04-01", "2016-07-01"])
    ap.add_argument("--screen-method", default="advi", choices=["advi", "nuts"])
    ap.add_argument("--no-verify", action="store_true", help="skip the small-NUTS verification")
    ap.add_argument("--samples", type=int, default=1500, help="ADVI posterior draws (screening)")
    ap.add_argument("--n-iter", type=int, default=20000, help="ADVI iterations (screening)")
    ap.add_argument("--ppc-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {label: (load_frozen_splits(label)[0]) for label in args.origins}
    temp_models = {}
    for label in args.origins:
        _, temp_train = load_frozen_splits(label)
        temp_models[label] = load_or_fit_temp_advi(temp_train, label, args.seed)

    frames = [(label, train) for label, train in splits.items()]
    print(f"[data] origins={args.origins}; training rows per split: "
          f"{[len(t) for _, t in frames]}; test horizon untouched.")

    candidates.assert_current_in_candidates("smart_home")
    current = candidates.SMART_HOME_CURRENT
    grids = candidates.SMART_HOME

    screening_kwargs = dict(
        scaler="minmax",
        scale_mode="individual",
        sigma_pool_type="individual",
        method=args.screen_method,
        n=args.n_iter,
        samples=args.samples,
        progressbar=args.progressbar,
    )

    def screening_kwargs_fn(label: str, _df):
        """Each origin transfers its own (ADVI) temperature posterior and its
        own time scale — identical to the study's per-origin setup."""
        return {
            "idata": temp_models[label].trace,
            "t_scale_params": temp_models[label].t_scale_params,
        }

    # Small-NUTS verification uses the primary origin only.
    primary_label = args.origins[0]
    nutpie_kwargs = dict(
        scaler="minmax",
        scale_mode="individual",
        sigma_pool_type="individual",
        method="nuts",
        samples=300,
        tune=300,
        chains=2,
        cores=2,
        nuts_sampler="nutpie",
        target_accept=0.9,
        progressbar=args.progressbar,
        idata=temp_models[primary_label].trace,
        t_scale_params=temp_models[primary_label].t_scale_params,
    )

    recommendations = {}
    reports = []

    for param in ["intercept_sd", "beta_sd", "yearly_order", "weekly_order"]:
        # Varying the Fourier order under prior_from_idata transfer is
        # inconsistent: the transferred prior's dimension is tied to the
        # *source* model's order.  Order selection therefore uses the
        # no-transfer arm (plain priors); the transferred shape is a separate
        # choice frozen in the study configuration.
        order_param = param in ("yearly_order", "weekly_order")
        factory = make_factory(tune_method=None) if order_param else make_factory(tune_method="prior_from_idata")
        print(f"\n=== {param} ===" + (" (no-transfer arm)" if order_param else ""))
        # Step 1: prior-predictive calibration (sd's only; plain priors).
        ppc_table = None
        if param in candidates.PRIOR_SCALE_PARAMS:
            plain_factory = make_factory(tune_method=None)
            ppc_table = bayesian.prior_predictive_sweep(
                plain_factory,
                frames[0][1],
                param,
                grids[param],
                samples=args.ppc_samples,
                seed=args.seed,
                progressbar=args.progressbar,
            )
            print(ppc_table.to_string(index=False))

        # Step 2: PSIS-LOO screening across origins (training windows only).
        screen_table, evidence = bayesian.run_loo_selection(
            factory,
            frames,
            param,
            grids[param],
            fit_kwargs=screening_kwargs,
            fit_kwargs_fn=screening_kwargs_fn,
            seed=args.seed,
            cache_dir=out_dir / "cache",
            cache_tag="plain" if order_param else "transfer",
            progressbar=args.progressbar,
        )
        print(screen_table.to_string(index=False))

        # Step 3: verify the top-2 with small NUTS (primary origin only).
        verify_table = None
        if not args.no_verify and len(screen_table) >= 2:
            top2 = bayesian.top_values_from_table(screen_table, evidence)
            try:
                verify_table = bayesian.verify_top_k(
                    factory, frames[:1], param, top2,
                    fit_kwargs=nutpie_kwargs, seed=args.seed,
                    progressbar=args.progressbar,
                )
                print("verification:\n", verify_table.to_string(index=False))
            except Exception as err:  # pragma: no cover - fall back to screening
                print(f"[verification failed for {param}; using the ADVI screening table: {err}]")
                verify_table = None

        # Selection rule (NUTS table when available, else ADVI screening).
        select_table = verify_table if verify_table is not None and not verify_table.empty else screen_table
        chosen, justification = bayesian.recommend(select_table, param, current[param])

        # Strategy B — leave-future-out CRPS (primary origin only); used as
        # a cross-check on the LOO pick (HYPERPARAMETER_SELECTION.md §3.1).
        cv_table = None
        try:
            cv_fit_kwargs = dict(screening_kwargs)
            cv_fit_kwargs["idata"] = temp_models[primary_label].trace
            cv_fit_kwargs["t_scale_params"] = temp_models[primary_label].t_scale_params
            cv_table = ts_cv.run_ts_cv(
                factory,
                frames[0][1],
                param,
                grids[param],
                fit_kwargs=cv_fit_kwargs,
                seed=args.seed,
                progressbar=args.progressbar,
            )
            cv_table.to_csv(out_dir / f"smart_home_{param}_ts_cv.csv", index=False)
            print("ts_cv:\n", cv_table.to_string(index=False))
        except Exception as err:  # pragma: no cover
            print(f"[ts_cv skipped for {param}: {err}]")

        # Cross-check flags: prior-predictive veto and CV disagreement.
        flags = []
        if ppc_table is not None and not ppc_table.empty:
            row = ppc_table[ppc_table[param] == chosen]
            if len(row) and row["verdict"].iloc[0] in ("too tight", "too wide"):
                flags.append(
                    f"PPC warning: prior-predictive coverage at {param}={chosen} is "
                    f"'{row['verdict'].iloc[0]}' ({row['coverage'].iloc[0]:.2f}); "
                    "the prior may be implausible for the scaled data."
                )
        if cv_table is not None and not cv_table.empty:
            cv_best = float(cv_table.iloc[0][param])
            if not np.isclose(cv_best, chosen):
                cv_chosen = cv_table.loc[cv_table[param] == chosen, "crps"]
                cv_chosen_txt = f"{cv_chosen.iloc[0]:.3f}" if len(cv_chosen) else "n/a"
                flags.append(
                    f"CV disagreement: leave-future-out CRPS prefers "
                    f"{param}={cv_best} (crps={cv_table.iloc[0]['crps']:.3f}) over "
                    f"the LOO pick {param}={chosen} (crps={cv_chosen_txt}); "
                    "report both (HYPERPARAMETER_SELECTION.md §3.1)."
                )
        if flags:
            justification = justification + " " + " ".join(flags)

        recommendations[param] = {
            "current": current[param],
            "recommended": chosen,
            "justification": justification,
            "candidates": grids[param],
        }

        bayesian.write_selection_report(
            out_dir,
            study="smart_home",
            param=param,
            ppc_table=ppc_table,
            screening_table=screen_table,
            verification_table=verify_table,
            recommended=chosen,
            current=current[param],
            justification=justification,
            extra={
                "provenance": bayesian.provenance(),
                "cross_check_flags": flags,
                "ts_cv_best": None if cv_table is None or cv_table.empty else float(cv_table.iloc[0][param]),
            },
        )

        # Strategy C — model-averaging weights from the screening pointwise elpd.
        pointwise = {n: e.pointwise for n, e in evidence.items()}
        weight_df = stacking.weight_table(pointwise, seed=args.seed)
        stacking.averaging_report(out_dir, "smart_home", param, weight_df)

        # Strategy D — robustness around the recommended value.
        try:
            rob_table, verdict = robustness.robustness_sweep(
                factory, frames, param, chosen,
                fit_kwargs=screening_kwargs, fit_kwargs_fn=screening_kwargs_fn,
                seed=args.seed,
                cache_dir=out_dir / "cache",
                cache_tag="plain" if order_param else "transfer",
                progressbar=args.progressbar,
            )
            robustness.write_robustness_report(out_dir, "smart_home", param, rob_table, verdict, chosen)
            print(f"robustness: {verdict}")
        except Exception as err:  # pragma: no cover
            print(f"[robustness skipped for {param}: {err}]")

    # Strategy E — full-Bayes hyperpriors on one appliance (demonstration).
    try:
        train0 = frames[0][1]
        appliance = train0["series"].unique()[0]
        model, trace, scale = full_bayes.fit_full_bayes(
            train0[train0["series"] == appliance],
            yearly_order=int(recommendations["yearly_order"]["recommended"]),
            weekly_order=int(recommendations["weekly_order"]["recommended"]),
            seed=args.seed,
            progressbar=args.progressbar,
        )
        full_bayes.report_summary(model, trace, out_dir, label=appliance.replace(" ", "_"))
        print(f"\n[full_bayes] posterior of the sd's on '{appliance}': "
              f"see full_bayes_{appliance.replace(' ', '_')}.md")
    except Exception as err:  # pragma: no cover
        print(f"[full_bayes skipped: {err}]")

    summary = {
        "study": "smart_home",
        "provenance": bayesian.provenance(),
        "candidates": grids,
        "recommendations": recommendations,
        "note": (
            "Selected on training windows only; no test-horizon data used. "
            "Freeze values into smart_home/config.py deliberately (see "
            "HYPERPARAMETER_SELECTION.md)."
        ),
    }
    (out_dir / "smart_home_recommendations.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\n[summary] recommendations -> {out_dir / 'smart_home_recommendations.json'}")
    for param, rec in recommendations.items():
        print(f"  {param}: {rec['current']} -> {rec['recommended']}")


if __name__ == "__main__":
    main()
