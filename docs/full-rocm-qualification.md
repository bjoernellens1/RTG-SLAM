# Full ROCm SLAM qualification (candidate)

**Status:** implementation candidate, not runtime-qualified. No new GPU results are claimed.
The previous mapper-only image remains the default target. Full adds the original
pinned CPU ORB-SLAM2 backend and Boost.Python binding on both vendor stacks.
The GPU algorithm is not replaced. CI must establish the build and runtime.

## Build (on the respective hosts)
```bash
git checkout rocm-full-qualification
docker build --target full -f docker/Dockerfile.rocm --build-arg ROCM_ARCH=gfx1151 -t rtg-full-rocm .
docker build -f docker/Dockerfile.cuda -t rtg-full-cuda .
```
Use gfx1201 for R9700, gfx1100 for W7900, or the actual architecture reported by rocminfo.
The CUDA build can run on a separate NVIDIA host at the **same commit**.
The full image is opt-in: it must not replace previously qualified mapper-only tags.

## Qualification
Download the original TUM fr1/desk dataset and keep it read-only. Mount an
existing empty parent results directory; each run creates a fresh child.
The loader stages its own writable directory. Use the original calibration.
Run a 120-frame prefix first, then the full sequence with --frames -1.
On each host run BOTH lanes: mapper (known poses + every tenth view withheld)
and slam (actual ICP+ORB tracking, no pose injection, no held-out views).

```bash
REV=$(git rev-parse HEAD)
IMAGE_ID=$(docker image inspect --format '{{.Id}}' rtg-full-rocm)
for lane in mapper slam; do
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -v "$TUM_DESK:/data:ro" -v "$RESULTS:/results" rtg-full-rocm \
    python scripts/qualification/run_tum.py --backend rocm --lane "$lane" \
    --dataset /data --config configs/tum/fr1_desk.yaml \
    --calibration configs/tum/dataset/fr1_desk.yaml --frames 120 \
    --source-revision "$REV" --image-id "$IMAGE_ID" --output "/results/rocm-$lane-120"
done
```

On the NVIDIA host, use rtg-full-cuda, --gpus all instead of AMD device flags,
--backend cuda, corresponding image ID and cuda output names. Copy the CUDA
results to the AMD host or another analysis host. Run with NumPy installed:

```bash
python scripts/qualification/compare_runs.py "$RESULTS/cuda-mapper-120" "$RESULTS/rocm-mapper-120" --output "$RESULTS/mapper-parity-120.json"
python scripts/qualification/compare_runs.py "$RESULTS/cuda-slam-120" "$RESULTS/rocm-slam-120" --output "$RESULTS/slam-parity-120.json"
```

The comparator rejects mismatched datasets, config, source, frame identities,
incomplete runs, corrupted artifact files, backend mismatch and invalid poses.
The initial numerical tolerances are proposed engineering gates, not measured
or paper-derived uncertainty. Repeat on full fr1/desk and fr3/long_office_household
with their original configs/calibration, three paired repetitions, and actual
wave32/wave64 hardware before widening public support claims.

## What the numbers mean
- Kernel parity: actual KNN, accumulation, rasterization and gradient tensors;
  integer outputs must match exactly. A single tiny fixture is necessary but not sufficient.
- Mapper parity: held-out view PSNR, SSIM, LPIPS, valid-depth ratio, depth MAE
  (conditional on valid rendered and measured depth). Same frozen poses and split.
- SLAM: matching ORB pose timestamps, nontrivial estimated motion, SE(3)-aligned
  translational ATE RMSE and ORB tracking-state success. Ground truth is used only
  for evaluation in this lane. Tracking/mapping run with original ICP/ORB.
- Timing: synchronized tracking/mapping stages, full online-loop throughput
  including checkpoint and cache overhead, final optimization time from loop start,
  and separate process wall time including initialization and offline outputs.
  Checkpoint metrics are observed-view diagnostics, not held-out time-to-quality.
  Compare their time axis only with matching configs. Final held-out quality is
  a separate evaluation for the mapper lane. Do not label memory as system VRAM:
  PyTorch allocated/reserved peaks are process allocator measurements.
- Fixed-view time-to-quality across checkpoints, additional real sensor datasets,
  per-kernel stress tests, geometry metrics and repeated-run variability are
  follow-ups. No claim of numerical equivalence follows from compilation alone.
