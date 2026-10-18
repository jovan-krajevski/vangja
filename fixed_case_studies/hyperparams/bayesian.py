"""The Bayesian hyperparameter-selection workflow (HYPERPARAMETER_SELECTION.md §2).

Steps implemented here:

1. ``prior_predictive_sweep`` — prior-predictive calibration of the prior
   scale (``utils.prior_predictive_coverage`` on the scaled ``[-2, 2]``
   window).  Cheap (``mapx`` graph fit + prior sampling).
2. ``run_loo_selection`` — coordinate-wise PSIS-LOO comparison of the
   candidate values on **training windows only** (pointwise elpd pooled
   across origins/units).  Screening fits use ADVI; the top-2 are verified
   with ``verify_top_k`` using small NUTS runs.
3. ``recommend`` — conservative selection rule (HYPERPARAMETER_SELECTION.md
   §2.2): keep the frozen value unless the best candidate beats it by more
   than ``dse_threshold`` dse; among values within the threshold prefer the
   smaller one (parsimony + the speed goal).

Every fit is seeded; wall time and parameter count are recorded so the
report can justify reducing ``n_changepoints`` on compute grounds as well.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.hyperparams.scoring import elpd_diff_se, loo_elpd_i, loo_table
from vangja.utils import prior_predictive_coverage

# Target prior-predictive coverage on the scaled data (utils docstring).
COVERAGE_LOW, COVERAGE_HIGH = -2.0, 2.0
COVERAGE_TARGET = (0.30, 0.60)


@dataclass
class CandidateEvidence:
    """Everything the workflow knows about one candidate value.

    ``pointwise`` is the PSIS-LOO pointwise elpd concatenated over origins;
    ``wall_time`` is the total screening fit time (seconds); ``n_params`` is
    the number of free model parameters (relevant for the changepoint
    speed argument).
    """

    value: float | int
    pointwise: np.ndarray
    wall_time: float = 0.0
    n_params: int = 0
    fitted: bool = True
    error: str | None = None

    @property
    def n_obs(self) -> int:
        return int(self.pointwise.size)


# ---------------------------------------------------------------------------
# Step 1 — prior-predictive calibration of the prior scale
# ---------------------------------------------------------------------------


def prior_predictive_sweep(
    model_factory,
    data: pd.DataFrame,
    param: str,
    candidates: list[float],
    samples: int = 400,
    seed: int = 42,
    fit_kwargs: dict | None = None,
    progressbar: bool = False,
) -> pd.DataFrame:
    """Coverage of the prior predictive inside the scaled [-2, 2] window.

    The ``model_factory`` must build the model **with plain priors**
    (``tune_method=None``) so the candidate sd is the actual prior scale.
    The graph is fit with ``method="mapx"`` (cheap) purely so that the
    prior predictive can be sampled.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(**{param: value}) -> TimeSeriesModel``.
    data : pd.DataFrame
        Training data (columns ``ds``, ``y``, optionally ``series``).
    param : str
        Hyperparameter name (the keyword the factory accepts).
    candidates : list[float]
        Candidate values.
    samples : int
        Prior-predictive draws per candidate.
    seed : int
        Seeded for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns ``[param, coverage, verdict]`` where verdict is one of
        "too wide", "target", "too tight" (HYPERPARAMETER_SELECTION.md §2.1).
    """
    fit_kwargs = {"method": "mapx", **(fit_kwargs or {})}
    rows = []
    for value in candidates:
        model = model_factory(**{param: value})
        kwargs = dict(fit_kwargs)
        kwargs.setdefault("random_seed", seed)
        kwargs.setdefault("progressbar", progressbar)
        model.fit(data, **kwargs)
        ppc = model.sample_prior_predictive(samples=samples, random_seed=seed)
        coverage = prior_predictive_coverage(ppc, COVERAGE_LOW, COVERAGE_HIGH)
        if coverage < COVERAGE_TARGET[0]:
            verdict = "too wide"
        elif coverage > COVERAGE_TARGET[1]:
            verdict = "too tight"
        else:
            verdict = "target"
        rows.append(
            {param: value, "coverage": round(float(coverage), 4), "verdict": verdict}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2 — PSIS-LOO comparison (screening)
# ---------------------------------------------------------------------------


def _data_fingerprint(frames: list[tuple[str, pd.DataFrame]]) -> str:
    h = hashlib.sha256()
    for label, df in frames:
        h.update(label.encode())
        h.update(
            pd.to_datetime(df["ds"]).sort_values().astype(str).str.cat(sep=",").encode()
        )
        h.update(df["series"].astype(str).sort_values().str.cat(sep=",").encode())
    return h.hexdigest()[:12]


def _fit_once(
    model_factory,
    data: pd.DataFrame,
    param: str,
    value: float,
    fit_kwargs: dict,
    seed: int,
    progressbar: bool,
):
    model = model_factory(**{param: value})
    kwargs = dict(fit_kwargs)
    # The seed/progressbar passed by the engine are defaults; per-call
    # fit_kwargs may override them (avoid duplicate-kwarg TypeErrors).
    kwargs.setdefault("random_seed", seed)
    kwargs.setdefault("progressbar", progressbar)
    t0 = time.perf_counter()
    model.fit(data, **kwargs)
    wall = time.perf_counter() - t0
    model.compute_log_likelihood()
    # Scalar parameter count (not variable count): what actually drives
    # sampling cost — e.g. n_changepoints adds one delta scalar per
    # changepoint, series_order adds two beta scalars per order.
    n_params = int(
        sum(
            int(np.prod([d if d else 1 for d in v.type.shape]))
            for v in model.model.free_RVs
        )
    )
    return model, wall, n_params


def run_loo_selection(
    model_factory,
    frames: list[tuple[str, pd.DataFrame]],
    param: str,
    candidates: list[float],
    *,
    fit_kwargs: dict | None = None,
    fit_kwargs_fn=None,
    seed: int = 42,
    cache_dir: Path | None = None,
    cache_tag: str = "",
    progressbar: bool = False,
    name_fn=None,
) -> tuple[pd.DataFrame, dict[str, CandidateEvidence]]:
    """Coordinate-wise PSIS-LOO comparison over candidates and origins.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(**{param: value}) -> TimeSeriesModel``.
    frames : list[tuple[str, pd.DataFrame]]
        ``(origin_label, training_df)`` pairs.  Only the training window of
        each origin is used; the test horizon is never passed in.
    param : str
        Hyperparameter name.
    candidates : list[float]
        Candidate values.
    fit_kwargs : dict or None
        Forwarded to ``model.fit`` (must include ``method`` etc.).  Used for
        every origin; for per-origin settings (e.g. each origin transfers its
        own context posterior) pass ``fit_kwargs_fn`` instead.
    fit_kwargs_fn : callable or None
        ``fit_kwargs_fn(origin_label, training_df) -> dict`` overriding
        ``fit_kwargs`` per origin (merged over it).
    cache_dir : Path or None
        If given, per-(param, value, origin) pointwise elpd is cached so the
        script resumes after interruption.
    name_fn : callable or None
        ``name_fn(param, value) -> str`` label for the table (default
        ``f"{param}={value}"``).

    Returns
    -------
    (table, evidence) : ranked LOO table and per-candidate evidence.
    """
    fit_kwargs = fit_kwargs or {}
    name_fn = name_fn or (lambda p, v: f"{p}={v}")
    fingerprint = _data_fingerprint(frames) if cache_dir is not None else ""

    evidence: dict[str, CandidateEvidence] = {}
    failures: list[str] = []
    for value in candidates:
        name = name_fn(param, value)
        pointwise_parts: list[np.ndarray] = []
        wall_total = 0.0
        n_params = 0
        fitted = True
        candidate_failures: list[str] = []
        for label, df in frames:
            kwargs = dict(fit_kwargs)
            if fit_kwargs_fn is not None:
                kwargs.update(fit_kwargs_fn(label, df))
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                method_tag = str(kwargs.get("method", "?"))
                key = f"{cache_tag}__{param}__{value}__{label}__seed{seed}__{method_tag}__{fingerprint}"
                npz = cache_dir / f"elpd_{key}.npz"
                meta = cache_dir / f"meta_{key}.json"
                if npz.exists() and meta.exists():
                    pointwise_parts.append(np.load(npz)["elpd_i"])
                    m = json.loads(meta.read_text())
                    wall_total += m["wall_time"]
                    n_params = m["n_params"]
                    continue
            try:
                model, wall, n_params = _fit_once(
                    model_factory, df, param, value, kwargs, seed, progressbar
                )
                elpd_i = loo_elpd_i(model)
                pointwise_parts.append(elpd_i)
                wall_total += wall
                if cache_dir is not None:
                    np.savez(npz, elpd_i=elpd_i)
                    meta.write_text(
                        json.dumps(
                            {"wall_time": wall, "n_params": n_params, "fitted": True}
                        )
                    )
            except Exception as err:
                fitted = False
                candidate_failures.append(f"{label}: {type(err).__name__}: {err}")
                continue
        if not pointwise_parts:
            error = "; ".join(candidate_failures) or "no origin fitted"
            evidence[name] = CandidateEvidence(
                value=value, pointwise=np.array([]), fitted=False, error=error
            )
            failures.append(f"{name} -> {error}")
            continue
        evidence[name] = CandidateEvidence(
            value=value,
            pointwise=np.concatenate(pointwise_parts),
            wall_time=wall_total,
            n_params=n_params,
            fitted=fitted,
            error="; ".join(candidate_failures) or None,
        )

    fitted_evidence = {n: e for n, e in evidence.items() if e.fitted}
    if not fitted_evidence and candidates:
        raise RuntimeError(
            f"No candidate fits succeeded for '{param}' (candidates={candidates}). "
            f"Failures:\n" + "\n".join(failures)
        )

    table = loo_table({n: e.pointwise for n, e in fitted_evidence.items()})
    table["wall_time_s"] = [evidence[n].wall_time for n in table["name"]]
    table["n_params"] = [evidence[n].n_params for n in table["name"]]
    return table, evidence


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def top_values_from_table(
    table: pd.DataFrame,
    evidence: dict[str, CandidateEvidence],
    k: int = 2,
) -> list[float | int]:
    """Top-k candidate *values* from a LOO table, preserving their type.

    Continuous hyperparameters (``intercept_sd``, ``beta_sd``, ...) are
    floats; discrete ones (``series_order``, ``n_changepoints``) are ints.
    Parsing the ``name`` string back with ``float()`` turns ``3`` into
    ``3.0``, which crashes the components (``np.empty`` shapes and
    ``range()`` need ints) — the ``evidence`` dict keeps the original
    typed values that were passed to the model factory.
    """
    names = table["name"].head(k).tolist()
    return [evidence[n].value for n in names]


def recommend(
    table: pd.DataFrame,
    param: str,
    current: float,
    *,
    dse_threshold: float = 1.0,
    prefer_smaller: bool = True,
) -> tuple[float | int, str]:
    """Conservative recommendation from a LOO table.

    Rule (HYPERPARAMETER_SELECTION.md §2.2):
    1. Candidates within ``dse_threshold`` dse of the best are statistically
       indistinguishable; if the frozen ``current`` value is among them, keep
       it ("no evidence to change").
    2. Otherwise, among the indistinguishable set, prefer the smallest value
       (parsimony / speed tie-break).
    3. If nothing is within the threshold, take the best candidate.

    Returns
    -------
    (value, justification) : recommended value and one-line justification.
    """
    if table.empty:
        return current, f"no candidate fits succeeded; keeping frozen value {current}"

    def _val(row) -> float | int:
        # Preserve the type: ``yearly_order=3`` -> int 3 (not float 3.0),
        # so the recommendation can be fed straight back into the factory.
        name = row["name"] if isinstance(row, pd.Series) else row
        v = float(str(name).split("=")[-1])
        return int(v) if v.is_integer() else v

    best = table.loc[table["elpd"].idxmax()]
    best_value = _val(best)
    best_elpd = float(best["elpd"])
    # Statistically indistinguishable set (elpd within dse_threshold of best).
    contenders = table[table["elpd_diff"] >= -dse_threshold]
    if contenders.empty:
        contenders = table.iloc[:1]
    contender_values = contenders["name"].map(_val)

    # Rule 1: keep the frozen value if it is in the indistinguishable set.
    if any(np.isclose(contender_values, current)):
        return current, (
            f"frozen value {current} is within {dse_threshold} dse of the best "
            f"({best_value}, elpd={best_elpd:.2f}); no evidence to change."
        )

    # Rule 2: smallest value among the indistinguishable set (only when the
    # best is not the unique contender).
    if prefer_smaller and len(contenders) > 1:
        smallest = contenders.loc[contender_values.idxmin()]
        return _val(smallest), (
            f"best={best_value} (elpd={best_elpd:.2f}); candidates within "
            f"{dse_threshold} dse are indistinguishable, choosing the smallest "
            f"value {_val(smallest)} (parsimony / sampling speed)."
        )

    # Rule 3: clear winner.
    return best_value, (
        f"best candidate by PSIS-LOO: {best_value} (elpd={best_elpd:.2f}), "
        f"beating the frozen value by more than {dse_threshold} dse."
    )


# ---------------------------------------------------------------------------
# Step 3 — NUTS verification of the top-2
# ---------------------------------------------------------------------------


def verify_top_k(
    model_factory,
    frames: list[tuple[str, pd.DataFrame]],
    param: str,
    top_values: list[float],
    *,
    fit_kwargs: dict,
    seed: int = 42,
    progressbar: bool = False,
) -> pd.DataFrame:
    """Small-NUTS refit + PSIS-LOO for the top candidates.

    ``fit_kwargs`` must select an MCMC method (e.g. ``method="nuts"`` with
    small ``samples``/``tune`` and ``nuts_sampler="nutpie"``).

    Returns
    -------
    pd.DataFrame
        Same schema as ``run_loo_selection``'s table, from the NUTS fits.
    """
    table, _ = run_loo_selection(
        model_factory,
        frames,
        param,
        top_values,
        fit_kwargs=fit_kwargs,
        seed=seed,
        cache_dir=None,
        progressbar=progressbar,
    )
    table["stage"] = "nuts_verification"
    return table


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_selection_report(
    out_dir: Path,
    *,
    study: str,
    param: str,
    ppc_table: pd.DataFrame | None,
    screening_table: pd.DataFrame,
    verification_table: pd.DataFrame | None,
    recommended: float,
    current: float,
    justification: str,
    extra: dict | None = None,
) -> Path:
    """Write the per-hyperparameter markdown report and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {study}: Bayesian selection of `{param}`",
        "",
        f"- **Frozen value (config.py):** `{current}`",
        f"- **Recommended value:** `{recommended}`",
        f"- **Justification:** {justification}",
        f"- **Evidence computed on training windows only** (no test horizon).",
        "",
        "## Screening (ADVI + PSIS-LOO)",
        "",
        _table_md(screening_table),
    ]
    if ppc_table is not None and not ppc_table.empty:
        lines += [
            "",
            "## Prior-predictive calibration (target 30-60 % coverage in [-2, 2])",
            "",
            _table_md(ppc_table),
        ]
    if verification_table is not None and not verification_table.empty:
        lines += [
            "",
            "## Verification (small NUTS + PSIS-LOO)",
            "",
            _table_md(verification_table),
        ]
    if extra:
        lines += ["", "## Notes", ""]
        for k, v in extra.items():
            lines.append(f"- **{k}:** {v}")
    lines.append("")
    text = "\n".join(lines)
    path = out_dir / f"{study}_{param}_selection.md"
    path.write_text(text)
    return path


def _table_md(df: pd.DataFrame) -> str:
    """Markdown table without requiring ``tabulate`` (optional dep)."""
    if df is None or df.empty:
        return "_no fits succeeded_"
    known = [
        "name",
        "elpd",
        "se",
        "elpd_diff",
        "dse",
        "weight",
        "wall_time_s",
        "n_params",
    ]
    if "name" in df.columns:
        cols = [c for c in known if c in df.columns]
    else:
        # Tables without a ``name`` column (e.g. prior-predictive sweeps)
        # keep all their columns.
        cols = list(df.columns)
    sub = df[cols].copy()
    for c in sub.columns:
        if c != "name" and pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].round(3)
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    body = [" | ".join(str(v) for v in row) for row in sub.itertuples(index=False)]
    return "\n".join([f"| {header} |", f"| {sep} |"] + [f"| {b} |" for b in body])


def provenance() -> dict:
    """Provenance dict recorded in every recommendation file."""
    return {
        "commit": common.commit_sha(),
        "environment": common.environment_info(),
    }
