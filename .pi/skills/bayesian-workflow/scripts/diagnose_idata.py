#!/usr/bin/env python3
"""Diagnose an MCMC/VI/MAP fit and return a verdict against project thresholds.

Input is either an ArviZ InferenceData file (``--idata trace.nc``, requires
arviz) or a diagnostics JSON (``--json diag.json``, see
references/thresholds.md for the schema). Writes
``diagnostics_<label>.json`` and ``diagnostics_<label>.md`` into ``--out`` when
given; otherwise prints the report. Exit: 0 pass/warn, 1 fail, 2 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "r_hat_max": 1.01,          # fail at >=
    "ess_bulk_min": 400,        # fail at <=
    "ess_tail_min": 400,        # fail at <=
    "bfmi_min": 0.3,            # fail at <=
    "divergences_max": 0,       # fail above
    "tree_depth_hits_max": 0,   # fail above
    "target_accept_min": 0.9,   # NUTS policy
    "map_gradient_norm_max": 1e-2,
}


def load_thresholds(path: Path | None) -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path is not None:
        thresholds.update(json.loads(path.read_text()))
    return thresholds


def _numeric_map(obj, reducer) -> dict[str, float]:
    """Flatten an arviz/xarray result into {variable_max/min: value}."""
    if obj is None:
        return {}
    out: dict[str, float] = {}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the project env
        return out

    def add(name, value):
        arr = np.asarray(value, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            out[name] = float(reducer(finite))

    if hasattr(obj, "data_vars"):
        for name, da in obj.data_vars.items():
            add(str(name), da.values)
        if out:
            return out
    if hasattr(obj, "children"):
        for child in obj.children.values():
            out.update(_numeric_map(child, reducer))
        return out
    add(getattr(obj, "name", "value"), obj)
    return out


def _sample_stat(idata, name):
    stats = getattr(idata, "sample_stats", None)
    if stats is None:
        try:
            stats = idata["sample_stats"]
        except Exception:
            return None
    try:
        da = stats[name]
    except Exception:
        return None
    try:
        return da.values
    except Exception:
        return None


def from_idata(path: Path) -> dict:
    try:
        import arviz as az
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            f"reading --idata requires arviz and numpy ({exc}); "
            "use --json with precomputed diagnostics instead"
        )
    idata = az.from_netcdf(str(path))
    payload: dict = {"label": path.stem, "source": str(path)}

    payload["r_hat"] = _numeric_map(az.rhat(idata), np.nanmax)
    payload["ess_bulk"] = _numeric_map(az.ess(idata, method="bulk"), np.nanmin)
    payload["ess_tail"] = _numeric_map(az.ess(idata, method="tail"), np.nanmin)
    try:
        bfmi = np.asarray(az.bfmi(idata), dtype=float)
        payload["bfmi"] = [float(v) for v in bfmi[np.isfinite(bfmi)]]
    except Exception as exc:  # pragma: no cover - depends on the trace contents
        payload["bfmi_error"] = str(exc)

    diverging = _sample_stat(idata, "diverging")
    if diverging is not None:
        payload["divergences"] = int(np.nansum(diverging))
    depth = _sample_stat(idata, "tree_depth")
    if depth is not None:
        depth = np.asarray(depth, dtype=float)
        if depth.size:
            payload["max_tree_depth"] = float(depth.max())
            payload["max_tree_depth_hits"] = int(np.sum(depth >= depth.max()))
    return payload


def evaluate(payload: dict, thresholds: dict) -> dict:
    method = str(payload.get("method", "unknown")).lower()
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str, severity: str = "fail") -> None:
        checks.append({
            "check": name,
            "status": "pass" if ok else severity,
            "detail": detail,
        })

    mcmc = method in ("nuts", "nutpie") or bool(
        payload.get("r_hat") or payload.get("ess_bulk") or payload.get("divergences") is not None
    )
    r_hat = payload.get("r_hat") or {}
    if mcmc:
        if r_hat:
            worst = max(r_hat.items(), key=lambda kv: kv[1])
            check("r_hat", worst[1] < thresholds["r_hat_max"],
                  f"max R-hat {worst[1]:.4f} at {worst[0]} (limit < {thresholds['r_hat_max']})")
        else:
            check("r_hat", False, "no R-hat values supplied", severity="warn")

        for key, threshold_key in (("ess_bulk", "ess_bulk_min"), ("ess_tail", "ess_tail_min")):
            values = payload.get(key) or {}
            if values:
                worst = min(values.items(), key=lambda kv: kv[1])
                check(key, worst[1] > thresholds[threshold_key],
                      f"min {key} {worst[1]:.1f} at {worst[0]} (limit > {thresholds[threshold_key]})")
            else:
                check(key, False, f"no {key} values supplied", severity="warn")

        bfmi = payload.get("bfmi") or []
        if bfmi:
            check("bfmi", min(bfmi) > thresholds["bfmi_min"],
                  f"min BFMI {min(bfmi):.3f} (limit > {thresholds['bfmi_min']})")
        else:
            check("bfmi", False, "no BFMI values supplied", severity="warn")

        div = payload.get("divergences")
        check("divergences", isinstance(div, int) and div <= thresholds["divergences_max"],
              f"divergences={div} (limit <= {thresholds['divergences_max']})")
        hits = payload.get("max_tree_depth_hits")
        if hits is not None:
            check("tree_depth", hits <= thresholds["tree_depth_hits_max"],
                  f"max-tree-depth hits={hits} (limit <= {thresholds['tree_depth_hits_max']})")
        else:
            check("tree_depth", False, "no max_tree_depth_hits supplied", severity="warn")

    if method in ("nuts", "nutpie"):
        ta = payload.get("target_accept")
        check("target_accept", isinstance(ta, (int, float)) and ta >= thresholds["target_accept_min"],
              f"target_accept={ta} (policy >= {thresholds['target_accept_min']})")

    if method in ("mapx", "map"):
        check("mcmc", True, "MAP point fit: MCMC diagnostics not applicable",
              severity="n/a")
        diag = payload.get("map_diagnostics") or {}
        if not diag:
            check("map_diagnostics", False,
                  "MAP fit has no map_diagnostics recorded", severity="warn")
        else:
            status = str(diag.get("status", "unknown")).lower()
            check("map_status", status in ("converged", "ok", "success"),
                  f"optimizer status={status!r}")
            grad = diag.get("gradient_norm")
            if grad is not None:
                check("map_gradient_norm", grad <= thresholds["map_gradient_norm_max"],
                      f"gradient norm={grad} (limit <= {thresholds['map_gradient_norm_max']})")

    if method in ("advi", "fullrank_advi", "svgd", "asvgd"):
        check("covariance", True,
              "VI source fit: mean-field/diagonal posterior; do not claim "
              "covariance preservation without a NUTS source (F-8)", severity="warn")
        check("mcmc", True, "VI fit: MCMC diagnostics not applicable", severity="n/a")

    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]
    verdict = "fail" if failed else ("warn" if warned else "pass")
    return {
        "label": payload.get("label", "fit"),
        "method": method,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "checks": checks,
        "n_fail": len(failed),
        "n_warn": len(warned),
        "verdict": verdict,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Convergence diagnostics — {report['label']}",
        "",
        f"- method: `{report['method']}`",
        f"- verdict: **{report['verdict'].upper()}** "
        f"({report['n_fail']} fail, {report['n_warn']} warn)",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['check']} | {check['status']} | {check['detail']} |")
    lines += [
        "",
        "Thresholds: R-hat < 1.01, ESS bulk/tail > 400, BFMI > 0.3, divergences = 0, "
        "tree-depth hits = 0, NUTS `target_accept` >= 0.9.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="diagnose_idata.py",
        description="Diagnose a fit against the project's convergence thresholds.",
        epilog="Example: diagnose_idata.py --json diag.json --out results/diag",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--idata", type=Path, help="ArviZ InferenceData file (.nc)")
    source.add_argument("--json", type=Path, help="precomputed diagnostics JSON")
    p.add_argument("--label", default=None, help="report label (default: file stem)")
    p.add_argument("--out", type=Path, default=None, help="directory for the reports")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="JSON file overriding individual thresholds")
    p.add_argument("--json-stdout", action="store_true", help="print the full report as JSON")
    args = p.parse_args(argv)

    try:
        thresholds = load_thresholds(args.thresholds)
        if args.json:
            if not args.json.exists():
                print(f"diagnostics JSON not found: {args.json}", file=sys.stderr)
                return 2
            payload = json.loads(args.json.read_text())
        else:
            if not args.idata.exists():
                print(f"InferenceData not found: {args.idata}", file=sys.stderr)
                return 2
            payload = from_idata(args.idata)
        if args.label:
            payload["label"] = args.label
        report = evaluate(payload, thresholds)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"diagnostics failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        label = str(report["label"]).replace("/", "_")
        (args.out / f"diagnostics_{label}.json").write_text(
            json.dumps(report, indent=2) + "\n")
        (args.out / f"diagnostics_{label}.md").write_text(render_markdown(report))

    if args.json_stdout:
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report), end="")
    return {"pass": 0, "warn": 0, "fail": 1}[report["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
