import sys
sys.path.insert(0, "/home/jovan/repos/vangja/.pi/state/calibration/snapshot/src")
import numpy as np, pandas as pd, pymc as pm, xarray as xr
from vangja.components import LinearTrend

def ID(post):
    ds={}
    for j,(k,v) in enumerate(post.items()):
        v=np.asarray(v); dims=("chain","draw")[:v.ndim]+tuple(f"x{j}_d{i}" for i in range(max(0,v.ndim-2)))
        ds[k]=xr.DataArray(v,dims=dims)
    return xr.DataTree.from_dict({"posterior": xr.Dataset(ds)})

data = pd.DataFrame({"ds": pd.date_range("2020-01-01",periods=30),"y":np.linspace(0,1,30),
                     "t":np.linspace(0,1,30),"series":"a"})
slope = np.random.default_rng(0).normal(0,1,size=(1,2000,1))
idata = ID({"lt_0 - slope": slope})   # source has NO delta (n_changepoints=0)
for mode in ["partial","individual"]:
    lt = LinearTrend(n_changepoints=0, tune_method="parametric", pool_type=mode)
    with pm.Model() as m:
        try:
            lt.definition(m, data, {}, None, idata)
            print(mode, "OK")
        except Exception as e:
            print(mode, "->", type(e).__name__, e)
