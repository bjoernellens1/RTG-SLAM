#!/usr/bin/env python3
"""Fail-closed CUDA/ROCm comparison of matching TUM runs. Thresholds are proposals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(cond, msg):
    if not cond:
        raise ValueError(msg)


def load_run(path):
    m = json.loads((path/"run.json").read_text())
    check(m["status"] == "completed", "run did not complete")
    for name, expected in m["files"].items():
        h = hashlib.sha256()
        with (path/name).open("rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                h.update(block)
        check(h.hexdigest() == expected, "artifact hash mismatch: "+name)
    t = json.loads((path/"timing.json").read_text())
    check(t["backend"] == m["backend"], "runtime backend mismatch")
    check(t["use_gt_pose"] == (m["lane"] == "mapper"), "pose mode mismatch")
    check(t["orb_active"] == (m["lane"] == "slam"), "ORB mode mismatch")
    es = np.load(path/"map/save_traj/pose_es.npy", allow_pickle=False)
    gt = np.load(path/"map/save_traj/pose_gt.npy", allow_pickle=False)
    n = len(t["frames"])
    check(es.shape == gt.shape == (n,4,4) and n >= 10, "pose shape mismatch")
    check(np.isfinite(es).all() and np.isfinite(gt).all(), "invalid pose")
    check(n == m["frames_processed"], "frame count mismatch")
    for k in ("online_s", "finalized_s"):
        check(np.isfinite(t[k]) and t[k]>0, "invalid timing "+k)
    return m,t,es,gt


def compare_arrays(a,b,rtol,atol):
    results={}
    with np.load(a,allow_pickle=False) as x, np.load(b,allow_pickle=False) as y:
        check(set(x.files)==set(y.files) and len(x.files)>8, "smoke tensor inventory mismatch")
        for k in sorted(x.files):
            u,v=x[k],y[k]
            check(u.shape==v.shape and u.dtype==v.dtype, "tensor shape/dtype mismatch: "+k)
            check(u.size>0 and np.isfinite(u).all() and np.isfinite(v).all(), "invalid tensor: "+k)
            delta=np.abs(u.astype(np.float64)-v.astype(np.float64))
            ok=np.allclose(u,v,rtol=rtol,atol=atol) if u.dtype.kind=="f" else np.array_equal(u,v)
            results[k]={"pass":bool(ok),"max_abs":float(delta.max())}
    return results


def ate(es,gt):
    """SE(3) translation RMSE; no scale fit."""
    x=es[:,:3,3].astype(float)
    y=gt[:,:3,3].astype(float)
    xc=x-x.mean(0); yc=y-y.mean(0)
    u,_,vh=np.linalg.svd(xc.T@yc)
    fix=np.eye(3); fix[-1,-1]=np.sign(np.linalg.det(u@vh))
    r=u@fix@vh
    return float(np.sqrt(np.mean(np.sum((xc@r-yc)**2,axis=1))))


def metric_rows(rows):
    keys=("psnr","ssim","lpips","depth_loss","valid_pixel_ratio")
    check(len(rows)>0, "empty quality rows")
    arr=np.asarray([[r[k] for k in keys] for r in rows],dtype=float)
    check(np.isfinite(arr).all(), "nonfinite metrics")
    return dict(zip(keys,map(float,arr.mean(0))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cuda",type=Path);p.add_argument("rocm",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rtol",type=float,default=1e-4);p.add_argument("--atol",type=float,default=1e-5)
    a=p.parse_args()
    ma,ta,ea,ga=load_run(a.cuda); mb,tb,eb,gb=load_run(a.rocm)
    check(ma["backend"]=="cuda" and mb["backend"]=="rocm","wrong backend ordering")
    check(ta["hip"] is None and bool(tb["hip"]), "not genuine CUDA/HIP runtime")
    for k in ("lane","dataset_sha256","config_sha256","source_revision","frames_requested","frames_processed"):
        check(ma[k]==mb[k],"unpaired "+k)
    check([f["timestamp"] for f in ta["frames"]]==[f["timestamp"] for f in tb["frames"]],"input frames differ")
    check(np.allclose(ga,gb,rtol=0,atol=1e-6),"ground truth differs")
    arrays=compare_arrays(a.cuda/"smoke.npz",a.rocm/"smoke.npz",a.rtol,a.atol)
    checks={"kernel_parity":all(r["pass"] for r in arrays.values())}
    report=dict(schema="rtg-paired-comparison/v1",lane=ma["lane"],dataset_sha256=ma["dataset_sha256"],
                source_revision=ma["source_revision"],arrays=arrays,checks=checks,
                thresholds=dict(rtol=a.rtol,atol=a.atol,psnr_db=0.3,ssim=0.01,lpips=0.02,
                                depth_loss_m=0.01,valid_pixel_ratio=0.02,ate_delta_m=0.02,
                                max_ate_m=0.15,min_orb_ok_ratio=0.90))
    for name,t,m in (("cuda",ta,ma),("rocm",tb,mb)):
        n=len(t["frames"])
        report[name]=dict(device=t["device"],online_fps=n/t["online_s"],
                         finalized_s=t["finalized_s"],process_wall_s=m["process_wall_s"],
                         peak_allocated_bytes=t["peak_allocated_bytes"],
                         peak_reserved_bytes=t["peak_reserved_bytes"],
                         tracking_p50_ms=float(np.median([f["tracking_s"] for f in t["frames"]])*1000),
                         mapping_p95_ms=float(np.percentile([f["mapping_s"] for f in t["frames"]],95)*1000),
                         observed_view_checkpoints=[f for f in t["frames"] if f["observed_view_quality"] is not None])
    if ma["lane"]=="mapper":
        check([r["timestamp"] for r in ta["heldout"]]==[r["timestamp"] for r in tb["heldout"]],"held-out views differ")
        qa,qb=metric_rows(ta["heldout"]),metric_rows(tb["heldout"])
        report["cuda"]["heldout"]=qa;report["rocm"]["heldout"]=qb
        for k,lim in (("psnr",0.3),("ssim",0.01),("lpips",0.02),("depth_loss",0.01),("valid_pixel_ratio",0.02)):
            checks["heldout_"+k]=abs(qa[k]-qb[k])<=lim
        checks["frozen_pose_cuda"]=bool(np.allclose(ea,ga,rtol=0,atol=1e-5))
        checks["frozen_pose_rocm"]=bool(np.allclose(eb,gb,rtol=0,atol=1e-5))
    else:
        expected=np.asarray([f["timestamp"] for f in ta["frames"]])
        for name,t,es,gt in (("cuda",ta,ea,ga),("rocm",tb,eb,gb)):
            stamps=np.asarray(t["orb_timestamps"])
            check(stamps.shape==expected.shape and np.allclose(stamps,expected,rtol=0,atol=1e-4),"ORB timestamps differ")
            score=ate(es,gt)
            ratio=float(np.mean([f["orb_tracking_ok"] is True for f in t["frames"]]))
            report[name].update(ate_rmse_m=score,orb_ok_ratio=ratio)
            checks[name+"_ate"]=score<=0.15
            checks[name+"_orb_ok"]=ratio>=0.90
            checks[name+"_nontrivial_motion"]=float(np.max(np.linalg.norm(es[:,:3,3]-es[0,:3,3],axis=1)))>0.05
        checks["ate_parity"]=abs(report["cuda"]["ate_rmse_m"]-report["rocm"]["ate_rmse_m"])<=0.02
    report["pass"]=all(checks.values())
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"pass":report["pass"],"checks":checks},indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__=="__main__":
    main()
