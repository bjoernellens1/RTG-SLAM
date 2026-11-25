#!/usr/bin/env python3
"""Run identical TUM RGB-D evidence on CUDA/ROCm; never mark a run qualified itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(root, calibration):
    h = hashlib.sha256()
    paths = [root / n for n in ("rgb.txt", "depth.txt", "groundtruth.txt")]
    for folder in ("rgb", "depth"):
        paths.extend(sorted(p for p in (root / folder).rglob("*") if p.is_file()))
    for p in paths:
        h.update(str(p.relative_to(root)).encode() + b"\0" + sha(p).encode())
    h.update(sha(calibration).encode())
    return h.hexdigest()


def checked(cmd, logfile, env):
    with logfile.open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cuda", "rocm"), required=True)
    p.add_argument("--lane", choices=("mapper", "slam"), required=True)
    p.add_argument("--frames", type=int, default=120, help="-1 for full sequence")
    p.add_argument("--save-step", type=int, default=40)
    p.add_argument("--source-revision", required=True, help="Exact git commit of the container source")
    p.add_argument("--image-id", required=True, help="Immutable Docker image digest or local image ID")
    a = p.parse_args()
    if a.frames != -1 and a.frames < 20:
        p.error("use at least 20 frames or -1")
    if a.save_step < 1:
        p.error("save-step must be positive")
    if a.output.exists():
        p.error("output must not exist (avoid mixing evidence)")
    root = a.dataset.resolve()
    config = a.config.resolve(strict=True)
    calib = a.calibration.resolve(strict=True)
    out = a.output.resolve()
    if any(c.isspace() for c in str(out)):
        p.error("use an output path without whitespace (upstream config uses shell cp)")
    out.mkdir(parents=True)
    # Upstream writes cameras.json and pose caches next to its dataset.
    # Stage only read-only inputs as symlinks; generated files stay in this run.
    staged = out / "input"
    staged.mkdir()
    for n in ("rgb", "depth", "rgb.txt", "depth.txt", "groundtruth.txt"):
        src = root / n
        if not src.exists():
            raise FileNotFoundError(src)
        (staged / n).symlink_to(src, target_is_directory=src.is_dir())
    shutil.copyfile(calib, staged / "config.yaml")
    data_hash = fingerprint(root, calib)
    # Flatten parent configs without modifying upstream files.
    from utils.config_utils import read_config
    cfg = vars(read_config(str(config))).copy()
    if cfg.get("type") != "TUM":
        p.error("TUM configuration required")
    cfg.update(parent="None", source_path=str(staged), save_path=str(out / "map"),
               frame_start=0, frame_step=0, frame_num=a.frames, save_step=a.save_step,
               mode="single process", device_list=[0], use_gt_pose=a.lane == "mapper",
               use_orb_backend=a.lane == "slam", eval=a.lane == "mapper", eval_llff=10,
               use_tensorboard=False, record_mem=False, quiet=False)
    config_file = out / "run.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    semantic_cfg = {k:v for k,v in cfg.items() if k not in ("source_path", "save_path")}
    config_hash = hashlib.sha256(yaml.safe_dump(semantic_cfg, sort_keys=True).encode()).hexdigest()
    env = dict(os.environ, EXPECT_BACKEND=a.backend, RTG_QUALIFICATION_JSON=str(out / "timing.json"),
               RTG_SMOKE_ARRAYS=str(out / "smoke.npz"), MPLBACKEND="Agg")
    manifest = dict(schema="rtg-tum-run/v1", status="running", backend=a.backend,
                    lane=a.lane, dataset_sha256=data_hash, config_sha256=config_hash,
                    frames_requested=a.frames, source_revision=a.source_revision, image_id=a.image_id)
    (out / "run.json").write_text(json.dumps(manifest, indent=2))
    checked([sys.executable, "scripts/smoke/rtg_extensions.py"], out / "smoke.log", env)
    if a.lane == "slam":
        checked([sys.executable, "scripts/smoke/rtg_orb.py"], out / "orb.log", env)
    start = time.perf_counter()
    checked([sys.executable, "slam.py", "--config", str(config_file)], out / "slam.log", env)
    wall_s = time.perf_counter() - start
    timing = json.loads((out / "timing.json").read_text())
    if timing["backend"] != a.backend or timing["use_gt_pose"] != (a.lane == "mapper"):
        raise ValueError("backend or pose mode mismatch")
    if timing["orb_active"] != (a.lane == "slam"):
        raise ValueError("ORB activation mismatch")
    es = np.load(out / "map/save_traj/pose_es.npy", allow_pickle=False)
    gt = np.load(out / "map/save_traj/pose_gt.npy", allow_pickle=False)
    frames = timing["frames"]
    if es.shape != gt.shape or es.shape != (len(frames), 4, 4) or len(frames) < 10:
        raise ValueError("trajectory/frame count mismatch")
    if not np.isfinite(es).all() or not np.isfinite(gt).all():
        raise ValueError("non-finite trajectory")
    if a.frames != -1:
        expected = a.frames - (a.frames + 9)//10 if a.lane == "mapper" else a.frames
        if len(frames) != expected:
            raise ValueError("truncated requested prefix")
    if a.lane == "slam":
        stamps = np.asarray(timing["orb_timestamps"], dtype=float)
        if stamps.shape != (len(frames),) or not np.allclose(stamps, [f["timestamp"] for f in frames], rtol=0, atol=1e-4):
            raise ValueError("ORB poses do not match input timestamps")
    if a.lane == "mapper" and not timing["heldout"]:
        raise ValueError("missing held-out views")
    manifest.update(status="completed", process_wall_s=wall_s, frames_processed=len(frames),
                    files={str(p.relative_to(out)):sha(p) for p in [
                        out/"timing.json",out/"smoke.npz",out/"smoke.log",out/"slam.log",
                        out/"map/save_traj/pose_es.npy",out/"map/save_traj/pose_gt.npy"
                    ] + ([out/"orb.log"] if a.lane == "slam" else [])})
    (out / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    # python scripts/qualification/run_tum.py from repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
