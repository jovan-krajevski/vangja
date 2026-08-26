"""Covariance-transfer validation (review F-8, PLAN P0-17 / P1-3).

    python fixed_case_studies/stocks/05_covariance_transfer.py

Compares joint (multivariate-Gaussian, ``prior_from_idata``) transfer with
marginal moment matching (``parametric``) using a **NUTS-fitted** source:

1. Fits the S&P 500 context model with NUTS (nutpie) on the requested
   confirmation origin and saves the posterior draws and the estimated
   covariance/correlation matrices of the transferred parameters.
2. Reports whether meaningful posterior dependence exists (largest absolute
   off-diagonal correlation, fraction of |corr| > 0.1).
3. Fits the frozen main target configuration twice on a stock subset —
   once with joint transfer, once with marginal transfer — and compares
   per-unit Relative MAE and calibration.

How the result is used (PROTOCOL.md §8): meaningful covariance + forecasting
gain => main contribution; covariance but no gain => secondary uncertainty
contribution; no measurable value => package capability only.  Mean-field
ADVI is never used for this comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import numpy as np
import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import (
    RESULTS_ROOT,
    build_target_model,
    fit_source,
    get_universe_for_origin,
    load_origin_data,
    rescale_dataset,
    seed_for,
)

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "protocol"


def covariance_summary(idata, var_names: list[str]) -> dict:
    """Estimated covariance/correlation of the transferred parameters."""
    arrays = []
    for v in var_names:
        if v in idata["posterior"]:
            arr = idata["posterior"][v].to_numpy()
            if arr.ndim == 2:  # (chain, draw) scalar parameter
                arr = arr.reshape(-1, 1)
            else:
                arr = arr.reshape(-1, arr.shape[-1])
            arrays.append(arr)
    if not arrays:
        return {"n_params": 0}
    X = np.concatenate(arrays, axis=1)
    cov = np.cov(X, rowvar=False)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "n_params": int(X.shape[1]),
        "max_abs_corr": float(np.abs(off).max()) if off.size else None,
        "frac_abs_corr_gt_01": float((np.abs(off) > 0.1).mean()) if off.size else None,
        "covariance_matrix": cov.tolist(),
        "correlation_matrix": corr.tolist(),
        "var_names": var_names,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2023-01-01")
    ap.add_argument("--max-stocks", type=int, default=30)
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    confirmation = common.load_confirmation_origins(PROTOCOL_DIR)
    frozen_origins = [o.strftime("%Y-%m-%d") for o in confirmation["origin"]]
    if args.origin not in frozen_origins:
        raise SystemExit(
            f"--origin {args.origin} is not a frozen confirmation origin"
        )

    out_dir = RESULTS_ROOT / "covariance_transfer" / args.origin
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_universe_for_origin(args.origin, require_frozen=True)
    smp_train, smp_test, stocks_train, stocks_test, availability = load_origin_data(
        args.origin, tickers, max_stocks=args.max_stocks, seed=common.BASE_SEED
    )
    stocks_train, stocks_test = rescale_dataset(smp_train, stocks_train, stocks_test)

    # 1. NUTS source fit with saved draws + covariance diagnostics.
    source_model = fit_source(
        smp_train, method="nuts", seed=common.BASE_SEED,
        progressbar=args.progressbar,
    )
    import pickle as _pkl; open(out_dir / "source_posterior.pkl", "wb").write(_pkl.dumps(source_model.trace))
    var_names = [
        "lt_0 - slope",
        "fs_0 - beta",
    ]
    cov_summary = covariance_summary(source_model.trace, var_names)
    cov_summary.update({"origin": args.origin, "availability": availability})
    common.save_json(out_dir / "source_covariance.json", cov_summary)
    print(
        f"Source covariance: max|corr| = {cov_summary['max_abs_corr']}, "
        f"frac(|corr|>0.1) = {cov_summary['frac_abs_corr_gt_01']}"
    )

    # 2. Joint vs marginal transfer on the frozen main configuration.
    rows = []
    for ci, (name, tune) in enumerate(
        [("joint_transfer", "prior_from_idata"), ("marginal_transfer", "parametric")]
    ):
        cell_cfg = cfg.CONFIGS["main"]
        model_cfg = cfg.StockConfig(
            name=name,
            description=f"{name} transfer on the frozen main configuration",
            tune_method=tune,
            lt_loss_factor=cell_cfg.lt_loss_factor,
            fs_loss_factor=cell_cfg.fs_loss_factor,
            lt_pool=cell_cfg.lt_pool,
            fs_pool=cell_cfg.fs_pool,
            lt_shrinkage=cell_cfg.lt_shrinkage,
            fs_shrinkage=cell_cfg.fs_shrinkage,
        )
        model = build_target_model(model_cfg)
        model.fit(
            stocks_train,
            scaler="maxabs",
            method="mapx",
            scale_mode="complete",
            sigma_pool_type="individual",
            t_scale_params=source_model.t_scale_params,
            idata=source_model.trace,
            random_seed=seed_for(0, ci),
            progressbar=args.progressbar,
        )
        yhat = model.predict(horizon=cfg.HORIZON_DAYS)
        unit_df = common.unit_metrics(
            model, stocks_test, yhat,
            origin=args.origin, config=name, stage="confirmation",
        )
        unit_df["seed"] = seed_for(0, ci)
        unit_df.to_csv(out_dir / f"units_{name}.csv", index=False)
        yhat.to_csv(out_dir / f"forecasts_{name}.csv", index=False)
        rows.append({name: common.aggregate_rel_mae(unit_df), "config": name})
        common.record_artifact(
            out_dir,
            study="stocks",
            stage="confirmation",
            origin=args.origin,
            config=name,
            model=model,
            unit_df=unit_df,
            future_df=yhat,
            extra={"covariance_summary": cov_summary},
        )

    summary = pd.DataFrame(rows).set_index("config")
    summary.to_csv(out_dir / "joint_vs_marginal_summary.csv")
    print(summary)
    print("Covariance-transfer comparison finished. Results in", out_dir)


if __name__ == "__main__":
    main()
