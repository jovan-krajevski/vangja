"""Calibration and MAP-diagnostics for the stocks finalists (F-15, P1-2, P1-4).

    python fixed_case_studies/stocks/06_calibration.py

On the requested confirmation origin and a stock subset, for the frozen
finalists (``main`` and ``no_transfer``):

1. **MAP multi-start diagnostics** — the frozen configurations are refit
   with several seeds; termination status, final objective, gradient norm
   and parameter spread across starts are recorded (MAP forecasts are
   point estimates, so optimizer diagnostics are what validates them).
2. **Interval coverage** — empirical coverage and width of the MAP
   residual-based intervals from ``predict_uncertainty`` on the test set.
3. **Posterior-predictive scoring** — the target models are refit with NUTS
   on the subset and CRPS / interval sharpness are computed from posterior
   predictive draws.

All outputs are saved as CSVs/JSON under ``results/calibration/``.
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


def crps_from_draws(draws: np.ndarray, y: np.ndarray) -> float:
    """CRPS via the energy form: E|X - y| - 0.5 * E|X - X'|."""
    n = draws.shape[0]
    if n < 2:
        return float("nan")
    abs_err = np.abs(draws - y).mean()
    pair = 0.0
    for i in range(n - 1):
        pair += np.abs(draws[i] - draws[i + 1 :]).sum()
    pair /= n * (n - 1) / 2
    return float(abs_err - 0.5 * pair)


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, width: float) -> dict:
    mask = ~(np.isnan(lo) | np.isnan(hi))
    y_, lo_, hi_ = y[mask], lo[mask], hi[mask]
    return {
        "n": int(mask.sum()),
        "coverage": float(((y_ >= lo_) & (y_ <= hi_)).mean()),
        "mean_width": float(np.mean(hi_ - lo_)),
        "nominal": width,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2023-01-01")
    ap.add_argument("--max-stocks", type=int, default=20)
    ap.add_argument("--configs", nargs="*", default=["main", "no_transfer"])
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    confirmation = common.load_confirmation_origins(PROTOCOL_DIR)
    frozen_origins = [o.strftime("%Y-%m-%d") for o in confirmation["origin"]]
    if args.origin not in frozen_origins:
        raise SystemExit(f"--origin {args.origin} is not a frozen confirmation origin")

    out_dir = RESULTS_ROOT / "calibration" / args.origin
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_universe_for_origin(args.origin, require_frozen=True)
    smp_train, smp_test, stocks_train, stocks_test, availability = load_origin_data(
        args.origin, tickers, max_stocks=args.max_stocks, seed=common.BASE_SEED
    )
    stocks_train, stocks_test = rescale_dataset(smp_train, stocks_train, stocks_test)
    source_model = fit_source(
        smp_train, method="nuts", seed=common.BASE_SEED,
        progressbar=args.progressbar,
    )

    map_diag_rows, coverage_rows, crps_rows = [], [], []
    for ci, name in enumerate(args.configs):
        cell_cfg = cfg.CONFIGS[name]

        # 1. MAP multi-start diagnostics.
        for seed in (0, 1, 2):
            model = build_target_model(cell_cfg)
            model.fit(
                stocks_train,
                scaler="maxabs",
                method="mapx",
                scale_mode="complete",
                sigma_pool_type="individual",
                t_scale_params=source_model.t_scale_params,
                idata=source_model.trace if cell_cfg.tune_method else None,
                random_seed=seed_for(0, 10 + ci * 3 + seed % 3),
                progressbar=args.progressbar,
            )
            diag = model.fit_info.get("map_diagnostics") or {}
            row = {"config": name, "seed": seed, "method": "mapx"}
            for k in ("success", "status", "nit", "nfev", "fun"):
                if k in diag:
                    row[k] = diag[k]
            row["gradient_norm_l2"] = diag.get("jac_l2")
            row["slope_shared"] = float(
                np.asarray(model.map_approx.get("lt_0 - slope_shared", [np.nan]))
                .ravel()[0]
            )
            map_diag_rows.append(row)

            # 2. Interval coverage from the MAP residual-based intervals.
            yhat_unc = model.predict_uncertainty(
                horizon=cfg.HORIZON_DAYS, interval_width=0.95
            )
            for code, series_name in model.groups_.items():
                if series_name in common.CONTEXT_SERIES:
                    continue
                merged = (
                    stocks_test[stocks_test["series"] == series_name][["ds", "y"]]
                    .merge(
                        yhat_unc[
                            ["ds", f"yhat_{code}", f"yhat_lower_{code}", f"yhat_upper_{code}"]
                        ],
                        on="ds",
                        how="inner",
                    )
                    .dropna()
                )
                if merged.empty:
                    continue
                cov = interval_coverage(
                    merged["y"].values,
                    merged[f"yhat_lower_{code}"].values,
                    merged[f"yhat_upper_{code}"].values,
                    0.95,
                )
                cov.update({"config": name, "series": series_name, "seed": seed})
                coverage_rows.append(cov)

        # 3. Posterior predictive (NUTS) for CRPS on the same subset.
        try:
            pp_model = build_target_model(cell_cfg)
            pp_model.fit(
                stocks_train,
                scaler="maxabs",
                method="nuts",
                scale_mode="complete",
                sigma_pool_type="individual",
                t_scale_params=source_model.t_scale_params,
                idata=source_model.trace if cell_cfg.tune_method else None,
                samples=500,
                tune=500,
                chains=2,
                cores=2,
                nuts_sampler="nutpie",
                target_accept=0.9,
                random_seed=seed_for(0, 50 + ci),
                progressbar=args.progressbar,
            )
            ppc = pp_model.sample_posterior_predictive()
            obs = ppc["posterior_predictive"]["obs"].to_numpy()
            # obs: (chain, draw, n_obs); aligned with model.data row order.
            # Posterior-predictive draws cover the training rows; CRPS is
            # computed in-sample per series.
            data = pp_model.data.reset_index(drop=True)
            scale_params = common.per_series_scale_params(pp_model)
            flat_draws = obs.reshape(-1, obs.shape[-1])
            for code, series_name in pp_model.groups_.items():
                if series_name in common.CONTEXT_SERIES:
                    continue
                idx = np.where(data["series"].values == series_name)[0]
                if len(idx) == 0:
                    continue
                y_min, y_max = scale_params[series_name]
                draws = (
                    flat_draws[:, idx] * (y_max - y_min) + y_min
                )  # unscale draws
                y_true = data["y"].values[idx] * (y_max - y_min) + y_min
                crps_rows.append(
                    {
                        "config": name,
                        "series": series_name,
                        "crps": crps_from_draws(draws, y_true),
                        "kind": "in_sample",
                    }
                )
        except Exception as err:
            common.record_failure(
                out_dir, study="stocks", stage="calibration",
                origin=args.origin, config=name, seed=0, error=err,
                traceback_text="see log",
            )
            print(f"Posterior-predictive step failed for {name}: {err}")

    pd.DataFrame(map_diag_rows).to_csv(out_dir / "map_diagnostics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "interval_coverage.csv", index=False)
    pd.DataFrame(crps_rows).to_csv(out_dir / "crps.csv", index=False)
    print("Calibration artifacts written to", out_dir)


if __name__ == "__main__":
    main()
