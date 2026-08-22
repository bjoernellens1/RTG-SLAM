#!/usr/bin/env python3
"""Import/API smoke for RTG-SLAM's exact ORB-SLAM2 Python binding."""
from __future__ import annotations

import json

import orbslam2

required = {
    "System",
    "Sensor",
    "TrackingState",
}
missing = sorted(name for name in required if not hasattr(orbslam2, name))
assert not missing, f"orbslam2 binding missing symbols: {missing}"
assert hasattr(orbslam2.Sensor, "RGBD")
assert hasattr(orbslam2.TrackingState, "OK")

print(
    json.dumps(
        {
            "module": "orbslam2",
            "capabilities": [
                "rtg_orbslam2_binding",
                "rtg_orbslam2_rgbd",
            ],
        },
        sort_keys=True,
    )
)
