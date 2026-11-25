# RTG-SLAM ROCm qualification and release plan

## Release scope

Prepare a community-maintained research port under
`bjoernellens1/rtg-slam-rocm`, default branch `main`, initial version
`v0.1.0`. Use the description “Real-time Gaussian Splatting SLAM on AMD GPUs
— ROCm/HIP port of RTG-SLAM” and topics `rocm`, `hip`, `amd`,
`gaussian-splatting`, `slam`, `robotics`, `3d-reconstruction`, `pytorch`,
and `rgbd`. Keep the upstream GPL-3.0 license, credits and citations. Audit
Graphdeco and every vendored or image-bundled dependency before distribution.

The release track is separate from CaRtGS-HIP. Share ROCm container patterns,
architecture discovery, wave32/wave64 testing, benchmark contracts and
qualification receipts where they preserve each method's implementation.

## Source and evidence

- The public `rocm-7.2` branch is at `45fbcaf81264dc3251203af305152163691d4488`.
  The ROCm quickstart and public HTTPS submodule updates are already there.
- Draft PR #2, `rocm-full-qualification`, is at
  `4372a07105e0a9531eca285ae8e699d3cecb20bf`. It adds opt-in full images with
  the original CPU ORB-SLAM2, GPU tensor parity and a paired TUM runner.
  The shared `camera_center` fix applies to CUDA and HIP. Keep those runs
  distinct from unmodified upstream.
- The earlier workflow built `gfx1151`, `gfx1201`, and `gfx90a` containers.
  Its CUDA build failed and the GPU qualification stages were skipped.
  The exact PR-head workflow also passed CPU qualification and the three
  mapper image builds. Its full ROCm image failed in Boost.Python because
  `pyconfig.h` was missing, and its CUDA build failed. Repair the binding's
  selected Python development include path and verify header and ABI
  discovery with a compile probe before rebuilding. Public release requires
  physical GPU evidence.
- Draft PR #2 revision `9941902c9b2b32c9a8b70f1bbf44be26b6759ebf`
  built the full ROCm image on the workstation, including the original
  ORB-SLAM2 bindings. Image digest
  `sha256:bc339f72c8afdfb18fe2c807eb14e5d5bbab5b7701c8ab28bef3cc4bdf2c7e5c`
  ran the Q0 probes on physical gfx1151 and exported rasterization, KNN,
  and accumulation tensors. Both Q1 lanes stopped on TUM fr1/desk frame 1.
  The mapper's temporary-point filter calls `knn_points`, and the image
  lacks PyTorch3D. Restore that GPU dependency and repeat both lanes.
  CUDA build, paired numerical equivalence, and completed mapping remain
  release gates.
- Draft PR #2 revision `4ab10c840334ad7b8541dee16c8ff375adf2c413`
  restores the original PyTorch3D call and pins its CUDA and HIP builds.
  The exact CUDA image built and passed the ORB import. Both physical
  Strix Halo Q1 lanes completed 120 frames using the functional overlay
  `sha256:ad9cb98373f0ebeffbd99f4ded6a532bc48bf19581fc3ee827e31192a6c0a41c`.
  Native SLAM recorded 120 successful ORB states and 1.65 cm aligned ATE
  RMSE. ICP reported success only for initialization. The controlled lane
  kept all supplied poses exact and evaluated 12 held-out views with
  7.50 dB PSNR. Repeat both lanes on the exact full images before claiming
  paired equivalence or full-sequence support.

[ROCm branch](https://github.com/bjoernellens1/RTG-SLAM/tree/rocm-7.2) ·
[Draft PR #2](https://github.com/bjoernellens1/RTG-SLAM/pull/2) ·
[Earlier CI matrix](https://github.com/bjoernellens1/RTG-SLAM/actions/runs/32574872820) ·
[Full candidate CI](https://github.com/bjoernellens1/RTG-SLAM/actions/runs/45675267848)

## Qualification

1. Build mapper and opt-in full images from the exact candidate SHA. Repair
   CUDA reference CI, run the CPU tests and verify dependency pins, image
   digests, and the loaded ROCm extensions. Use the active cluster kubeconfig
   for authorized A100 experiments. Check job admission and capacity first.
2. Compare identical serialized kernel inputs across CUDA and HIP. Require
   relative tolerance `1e-4` for floats and exact integer agreement. Run
   Strix Halo first, then measure the same 120-frame TUM fr1/desk prefix on
   A100. Use the paired runner and retain raw tensors and gradients.
3. Run both lanes from the qualification guide. The controlled mapper lane
   injects the same calibrated poses and withholds the same evaluation views.
   The SLAM lane activates original ICP and ORB-SLAM2 and estimates motion
   from RGB-D. Confirm tracking activation and source-timestamp association.
4. Run three paired repetitions on complete TUM fr1/desk and fr3/office.
   Use these frozen engineering gates.

   | Metric | Gate |
   | --- | --- |
   | PSNR difference | ≤ 0.3 dB |
   | SSIM difference | ≤ 0.01 |
   | LPIPS difference | ≤ 0.02 |
   | Depth MAE difference | ≤ 1 cm |
   | Valid-depth coverage difference | ≤ 2 percentage points |
   | ATE difference | ≤ 2 cm |
   | Absolute ATE RMSE | ≤ 15 cm |
   | ORB tracking success | ≥ 90% |

5. Repeat on Radeon W7900 `gfx1100`, R9700 `gfx1201`, and a physical
   Instinct wave64 device, initially MI200 `gfx90a`. Record each measured
   architecture separately. Record synchronized tracking and mapping latency,
   ingest throughput, drain time, process allocator peaks and device memory.
6. Add fixed held-out checkpoint rendering and time-to-quality using the
   shared gs-slam-bench interfaces. Keep observed-view diagnostics separate
   from held-out quality. Archive complete terminal artifacts and failed
   runs alongside exact commands, source/image/data/config hashes and raw
   timings. Publish redistributable artifacts only.

[Candidate qualification commands](../../full-rocm-qualification.md)

## Redistribution inventory

The verified ROCm image contains the depth rasterizer and its license under
`/opt/rtg-cuda-utils/diff-gaussian-rasterizer-depth` from RTG CUDA utilities
revision `24dcf0aec00f35229f247ac206a5c0b141e15991`. Its separate research
license limits use and requires the complete license and existing notices
to accompany covered distributions. The candidate PyTorch3D HIP source
uses a BSD license with its own notice requirements. Record each component
in the release notice bundle. [Pinned rasterizer license](https://github.com/CJAPPLE5/RTG-SLAM-cuda_utils/blob/24dcf0aec00f35229f247ac206a5c0b141e15991/diff-gaussian-rasterizer-depth/LICENSE.md) · [Pinned PyTorch3D license](https://github.com/facebookresearch/pytorch3d/blob/b73d735ecf194c31de812feffef3a55cc3726128/LICENSE)

The source inventory and image notices still need confirmation for the
RTG utility and Simple-KNN code, the original ORB frontend and vocabulary,
the CUDA PyTorch3D package, base-image and system packages, and any model
weights or fixtures attached to a release.

## Release and community launch

- Gate the first mapper preview on verified Strix Halo runtime smoke, a
  complete controlled mapping sequence, and the license audit. Label each
  supported architecture and publish the limitations and reproducibility
  guide. Add a full-SLAM support claim only after the independent end-to-end
  and paired CUDA/HIP gates pass. Keep the draft PR status until reviewed.
- After release authorization, create the standalone public repository
  from the verified revision, set the repository metadata, publish
  `v0.1.0` and attach the audited result package. Preserve versioned
  mapper-only and full-image tags.
- Coordinate upstream RTG-SLAM and AMD Moat submissions first. The Moat
  issue should identify the existing mapper-core port, measured GPUs,
  exact results, known gaps and wave32/wave64 test requests. Follow with
  AMD Developer Hub, r/ROCm, r/StrixHalo, r/GaussianSplatting, r/robotics,
  Awesome ROCm and LinkedIn when release claims have evidence. Tailor each
  message to the audience and link the evidence and contribution guide.
- Public creation, release publication and outreach are execution steps
  that require explicit authorization. This plan update records them.

[AMD Moat](https://github.com/AMD-Ecosystem/moat) ·
[Upstream RTG-SLAM](https://github.com/MisEty/RTG-SLAM)
