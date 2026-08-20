# RTG-SLAM ROCm portability candidate

Pinned scientific identity:

- RTG-SLAM: `49dada148551961e2dee38040a4f760fa1b0f01d`
- `RTG-SLAM-cuda_utils` parent gitlink: `24dcf0aec00f35229f247ac206a5c0b141e15991`
- `RTG-SLAM-PYBIND` gitlink: `6be964f432f26f8c1802ab579190a0f9081e6147`
- `RTG-SLAM-BACKEND` gitlink: `9cb671553e72782f354a91825187cdcfc83e6f30`
- ROCm target: 7.2.4

The controlled mapper-core ROCm image intentionally does **not** build/port the
ORB frontend: the ResearchFlow controlled wrapper disables it and injects the
canonical frozen trajectory. The ROCm qualification instead covers the three
mapping extensions shipped inside the exact `RTG-SLAM-cuda_utils` gitlink:
RTG cuda-utils, the depth Gaussian rasterizer, and simple-knn.

Portability edits are backend/build-only. PyTorch `CUDAExtension` performs HIP
translation; the patch routes cooperative groups to HIP, removes CUDA-only
convenience headers, and normalizes kernel-launch spelling. No renderer/map
algorithm is replaced.
