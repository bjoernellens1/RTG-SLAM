#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

OCI = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA = re.compile(r"^[0-9a-f]{64}$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--digest", required=True)
    p.add_argument("--gpu-arch", required=True)
    p.add_argument("--amd-port-audit-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not OCI.fullmatch(a.digest):
        raise SystemExit("--digest must be OCI sha256")
    if not SHA.fullmatch(a.amd_port_audit_sha256):
        raise SystemExit("--amd-port-audit-sha256 must be 64 hex")
    payload = {
        "schema_version": "researchflow-container-qualification/v1",
        "method": "rtg_slam",
        "backend": "rocm",
        "backend_version": "7.2.4",
        "image": a.image,
        "digest": a.digest,
        "gpu_arch": a.gpu_arch,
        "status": "qualified",
        "base_image_digest": "sha256:9c9592175fece788d6c0b86059012f49b568dc95c98c13879dfdf89c30342559",
        "amd_port_audit_sha256": a.amd_port_audit_sha256,
        "moat_revision": "f69bb67d70e7a47af00095dab7029c230a30be73",
        "upstream_revision": "49dada148551961e2dee38040a4f760fa1b0f01d",
        "rtg_cuda_utils_revision": "24dcf0aec00f35229f247ac206a5c0b141e15991",
        "rtg_pybind_revision": "6be964f432f26f8c1802ab579190a0f9081e6147",
        "rtg_orb_backend_revision": "9cb671553e72782f354a91825187cdcfc83e6f30",
        "amd_gaussian_splatting_reference_revision": "dfb1dd1cbcc508e08c7395b27953620121dad1fc",
        "capabilities": ["rtg_cuda_utils_gpu", "rtg_depth_rasterizer_gpu", "simple_knn_gpu"],
        "scope": "controlled_mapper_core",
        "github": {
            "repository": os.getenv("GITHUB_REPOSITORY"),
            "sha": os.getenv("GITHUB_SHA"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["qualification_id"] = hashlib.sha256(raw).hexdigest()
    Path(a.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
