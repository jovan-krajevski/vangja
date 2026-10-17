import sys
sys.path.insert(0, "/home/jovan/repos/vangja/.pi/state/calibration/snapshot/src")
import numpy as np, pandas as pd, pymc as pm, xarray as xr
import vangja
print("vangja imported from:", vangja.__file__)
from vangja.components import LinearTrend, FourierSeasonality
import pymc_extras as pmx

rng = np.random.default_rng(0)

def ID(posterior: dict):
    ds = {}
    for j, (k, v) in enumerate(posterior.items()):
        v = np.asarray(v)
        dims = ("chain", "draw")[:v.ndim] + tuple(f"x{j}_d{i}" for i in range(max(0, v.ndim - 2)))
        ds[k] = xr.DataArray(v, dims=dims)
    return xr.DataTree.from_dict({"posterior": xr.Dataset(ds)})

def make_data(series=("a",)):
    return pd.concat([pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=30),
        "y": np.linspace(0.0, 1.0, 30), "t": np.linspace(0.0, 1.0, 30),
        "series": s}) for s in series], ignore_index=True)

# ---------------- P1: loss potential sign per pooling mode ----------------
slope_samples = rng.normal(0.0, 1.0, size=(1, 2000))
idata = ID({"lt_0 - slope": slope_samples, "lt_0 - delta": np.zeros((1, 2000, 1))})
print("\n== P1: LinearTrend loss_factor_for_tune=1 potential at slope=1 (source mean=0) ==")
for mode in ["complete", "partial", "individual"]:
    lt = LinearTrend(n_changepoints=0, tune_method="parametric", pool_type=mode,
                     loss_factor_for_tune=1.0)
    with pm.Model() as m:
        lt.definition(m, make_data(("a", "b") if mode != "complete" else ("a",)), {}, None, idata)
    pot = m.potentials[0]
    ng = 2 if mode != "complete" else 1
    if mode == "complete":
        p0 = {"lt_0 - slope": np.array(0.0)}
        p1 = {"lt_0 - slope": np.array(1.0)}
    elif mode == "partial":
        p0 = {"lt_0 - slope_shared": np.array(0.0), "lt_0 - slope_z_offset": np.zeros(ng), "lt_0 - slope_sigma": np.array(0.1)}
        p1 = {"lt_0 - slope_shared": np.array(1.0), "lt_0 - slope_z_offset": np.zeros(ng), "lt_0 - slope_sigma": np.array(0.1)}
    else:
        p0 = {"lt_0 - slope": np.zeros(ng)}
        p1 = {"lt_0 - slope": np.ones(ng)}
    g0 = pot.eval({m.named_vars[k]: v for k, v in p0.items()})
    g1 = pot.eval({m.named_vars[k]: v for k, v in p1.items()})
    print(f"  pool={mode:10s} free_RVs={[rv.name for rv in m.free_RVs]}")
    print(f"  pool={mode:10s} potential(slope=1)-potential(slope=0) = {float(g1)-float(g0): .3f}  (name: {pot.name})")

# ---------------- P2: covariance preservation prior_from_idata ----------------
N = 20000
z = rng.normal(size=(N, 4))
L = np.array([[2.0, 0, 0, 0],
              [1.8, 0.872, 0, 0],
              [0.0, 0.0, 1.5, 0],
              [0.0, 0.0, 1.2, 0.9]])
b = z @ L.T   # covariance = L L^T
src = b[None, :, :]  # (chain, draw, 4)
beta_key = "fs_0 - beta(p=365.25,n=2)"
idata_b = ID({beta_key: src})
src_cov = np.cov(src[0], rowvar=False)
src_sd = np.sqrt(np.diag(src_cov))
src_corr = src_cov / np.outer(src_sd, src_sd)
print("\n== P2: prior_from_idata target prior covariance vs source posterior ==")
print("source sd:", np.round(src_sd, 4))
print("source corr[0,1], corr[0,2], corr[0,3]:",
      round(src_corr[0,1],4), round(src_corr[0,2],4), round(src_corr[0,3],4))
for mode in ["complete", "partial", "individual"]:
    fs = FourierSeasonality(365.25, 2, tune_method="prior_from_idata", pool_type=mode)
    fs._assign_model_idx({})
    with pm.Model() as m:
        pv = {v: f"prior_{v}" for v in fs._get_prior_var_names()}
        priors = pmx.utils.prior.prior_from_idata(idata_b, name="priors", **pv)
        fs.definition(m, make_data(("a", "b")), {}, priors, idata_b)
        beta = m.named_vars[beta_key]
        dr = pm.draw(beta, draws=N, random_seed=1)
    dr2 = dr.reshape(-1, 4)
    cov = np.cov(dr2, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    print(f"  pool={mode:10s} target sd={np.round(sd,4)}  var_ratio={np.round(np.diag(cov)/np.diag(src_cov),3)}")
    print(f"  pool={mode:10s} target corr[0,1]={corr[0,1]:.4f}  corr[0,2]={corr[0,2]:.4f}  corr[0,3]={corr[0,3]:.4f}")
    # one-sided loss potential is inactive (loss_factor default 0): fine

# LinearTrend individual prior_from_idata variance inflation
slope_key = "lt_0 - slope"
s_samples = rng.normal(0.0, 2.0, size=(1, N, 1))
idata_s = ID({slope_key: s_samples, "lt_0 - delta": rng.normal(0, 0.1, size=(1, N, 2))})
lt = LinearTrend(n_changepoints=2, tune_method="prior_from_idata", pool_type="individual")
lt._assign_model_idx({})
with pm.Model() as m:
    pv = {v: f"prior_{v}" for v in lt._get_prior_var_names()}
    priors = pmx.utils.prior.prior_from_idata(idata_s, name="priors", **pv)
    lt.definition(m, make_data(("a", "b")), {}, priors, idata_s)
    dr = pm.draw(m.named_vars[slope_key], draws=N, random_seed=2)
print(f"  LinearTrend individual: source var={np.var(s_samples):.3f} target var={np.var(dr):.3f} ratio={np.var(dr)/np.var(s_samples):.3f}")

# ---------------- P3: estimator actually used (mean vs mode) ----------------
skew = rng.exponential(1.0, size=(1, 1, 20000))
idata_sk = ID({slope_key: skew})
lt = LinearTrend(n_changepoints=0, tune_method="parametric", pool_type="complete"); lt.model_idx = 0
mu, sd = lt._get_slope_params_from_idata(idata_sk)
h, e = np.histogram(skew.ravel(), bins=400, density=True)
mode = 0.5 * (e[h.argmax()] + e[h.argmax()+1])
flat = skew.ravel()
print("\n== P3: parametric transfer location estimate on skewed posterior ==")
print(f"  returned loc={mu:.5f}  sample mean={flat.mean():.5f}  returned sd={sd:.5f}  histogram-mode={mode:.5f}")
print(f"  returned loc equals mean: {np.isclose(mu, flat.mean())}   loc-mode = {mu-mode:+.5f}")

# P3b: single-draw (MAP) posterior -> zero scale
one = ID({slope_key: np.array([[[0.5]]])})
mu1, sd1 = lt._get_slope_params_from_idata(one)
try:
    logp = pm.logp(pm.Normal.dist(mu1, sd1), np.array(mu1)).eval()
except Exception as e:
    logp = f"{type(e).__name__}: {e}"
print(f"\n== P3b: parametric transfer from 1-draw (MAP-like) posterior ==")
print(f"  returned (loc, sd) = ({mu1}, {sd1}); Normal(loc, sd) logp at loc = {logp}")
