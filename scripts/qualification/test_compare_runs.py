"""CPU-only tests: no fake GPU pass or simulated benchmark results."""
import numpy as np
import pytest

from compare_runs import ate, compare_arrays, metric_rows


def poses(x):
    p=np.repeat(np.eye(4)[None],len(x),axis=0)
    p[:,:3,3]=x
    return p


def test_ate_rigid_transform():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,2.]])
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    assert ate(poses(x),poses(x@rot + np.array([2.,3.,4.]))) < 1e-12
    assert ate(poses(x),poses(x*2)) > 0.1  # do not hide scale error


def pair(tmp_path, delta=0., int_delta=0, nan=False, missing=False):
    a=tmp_path/"cuda.npz"; b=tmp_path/"rocm.npz"
    x={f"f{i}":np.array([1.,2.,3.],dtype=np.float32) for i in range(9)}
    x["indices"]=np.array([0,1,2],dtype=np.int32)
    y={k:v.copy() for k,v in x.items()}
    y["f0"][0]+=delta
    y["indices"][0]+=int_delta
    if nan: y["f0"][0]=np.nan
    if missing: del y["f1"]
    np.savez(a,**x); np.savez(b,**y)
    return a,b


def test_tensor_parity(tmp_path):
    a,b=pair(tmp_path,delta=1e-6)
    assert all(r["pass"] for r in compare_arrays(a,b,1e-4,1e-5).values())


def test_float_mismatch(tmp_path):
    a,b=pair(tmp_path,delta=0.1)
    assert not compare_arrays(a,b,1e-4,1e-5)["f0"]["pass"]


def test_integer_mismatch(tmp_path):
    a,b=pair(tmp_path,int_delta=1)
    assert not compare_arrays(a,b,1e-4,1e-5)["indices"]["pass"]


@pytest.mark.parametrize("kwargs",[{"nan":True},{"missing":True}])
def test_reject_invalid_evidence(tmp_path,kwargs):
    a,b=pair(tmp_path,**kwargs)
    with pytest.raises(ValueError):
        compare_arrays(a,b,1e-4,1e-5)


def test_reject_invalid_quality():
    with pytest.raises(ValueError):
        metric_rows([])
    with pytest.raises(ValueError):
        metric_rows([dict(psnr=np.nan,ssim=1.,lpips=0.,depth_loss=0.,valid_pixel_ratio=1.)])
