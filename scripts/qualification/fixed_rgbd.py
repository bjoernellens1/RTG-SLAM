"""Read a sealed fixed-calibration mapper packet as RTG-SLAM camera records."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _fields(value, fields, label):
    _require(isinstance(value, dict) and value.keys() == set(fields.split()), f"Unexpected {label} fields")


def _read_media(root, media, frame_id, size):
    _fields(media, "rgb depth masks", f"{frame_id} media")
    images, paths = {}, {}
    for kind, mode in (("rgb", "RGB"), ("depth", "I;16"), ("masks", "L")):
        entry = media[kind]
        _fields(entry, "path sha256", f"{frame_id} {kind}")
        relative = f"{kind}/{frame_id}.png"
        path = root / relative
        _require(entry["path"] == relative and path.resolve() == path, f"Unexpected media path: {relative}")
        data = path.read_bytes()
        _require(_sha(data) == entry["sha256"], f"Incorrect media digest: {relative}")
        with Image.open(BytesIO(data)) as image:
            if kind == "depth":
                # Pillow versions may expose 16-bit grayscale PNG as mode I.
                valid_mode = (
                    image.mode in ("I", "I;16")
                    and data[12:16] == b"IHDR"
                    and data[24:26] == b"\x10\x00"
                )
            else:
                valid_mode = image.mode == mode
            _require(image.format == "PNG" and image.size == size and valid_mode,
                     f"Unexpected PNG encoding or geometry: {relative}")
            images[kind] = image.copy()
        paths[kind] = str(path)

    mask = np.asarray(images["masks"])
    depth = np.asarray(images["depth"], dtype=np.float32)
    unsupported = mask == 0
    _require(np.all((mask == 0) | (mask == 255)) and not np.any(depth == 65535)
             and not np.any(depth[unsupported]) and not np.any(np.asarray(images["rgb"])[unsupported]),
             f"Unexpected mask or depth samples: {frame_id}")
    return images["rgb"], Image.fromarray(depth / 1000), paths


def load_fixed_rgbd(manifest_path, *, expected_manifest_sha256, camera_info_type=None):
    """Return the complete, externally pinned mapper packet in capture order.

    The default constructor is RTG-SLAM's CameraInfo. Its legacy `pose_gt`
    slot carries the packet's supplied causal camera pose for fixed-pose
    mapping. Depth is stored as a float PIL raster in meters, with zeros
    preserved, and `depth_scale` records the original millimeter conversion.
    """
    manifest = Path(manifest_path).resolve()
    root = manifest.parent
    content = manifest.read_bytes()
    _require(_sha(content) == expected_manifest_sha256, "Incorrect manifest digest")
    packet = json.loads(content)
    _fields(packet, "schema role scene_id source_prefix_sha256 transform_version pose_axes pose_protocol "
            "depth_units_per_meter calibration fixed_calibration_sha256 associations_sha256 frames input_sha256",
            "manifest")
    _require(packet["schema"] == "sceneweave-cartgs-fixed-calibration-input-v1"
             and packet["role"] == "mapper_input" and packet["transform_version"] == "registered-fixed-k00-v1"
             and packet["pose_axes"] == "opencv_camera_to_zup_world"
             and packet["pose_protocol"] == "frame0_relative_arkit_vio"
             and type(packet["depth_units_per_meter"]) is int and packet["depth_units_per_meter"] == 1000,
             "Unsupported mapper input schema, calibration transform, pose protocol, or depth units")
    source_prefix = packet["source_prefix_sha256"]
    _require(isinstance(source_prefix, str) and re.fullmatch("[0-9a-f]{64}", source_prefix),
             "Invalid source prefix digest")
    _require(_json_sha({k: v for k, v in packet.items() if k != "input_sha256"}) == packet["input_sha256"],
             "Incorrect canonical input digest")
    calibration = packet["calibration"]
    _fields(calibration, "width height K frame_id", "calibration")
    _require(_json_sha(calibration) == packet["fixed_calibration_sha256"], "Incorrect calibration digest")
    width, height = calibration["width"], calibration["height"]
    k = np.asarray(calibration["K"], dtype=np.float64)
    _require(type(width) is int and type(height) is int and min(width, height) > 0
             and k.shape == (3, 3) and np.isfinite(k).all(), "Invalid fixed calibration dimensions or matrix")
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    _require(min(fx, fy) > 0 and 0 <= cx < width and 0 <= cy < height
             and np.array_equal(k, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
             and calibration["frame_id"] == "frame_000000", "Invalid fixed calibration or frame-0 binding")
    fovx, fovy = 2 * math.atan(width / (2 * fx)), 2 * math.atan(height / (2 * fy))

    frames = packet["frames"]
    _require(isinstance(frames, list) and frames, "Expected a nonempty mapper frame list")
    associations = root / "associations.txt"
    _require(associations.resolve() == associations, "Unexpected association path")
    association_bytes = associations.read_bytes()
    _require(_sha(association_bytes) == packet["associations_sha256"], "Incorrect association digest")
    association_rows = association_bytes.decode().splitlines()
    _require(len(association_rows) == len(frames), "Incorrect association frame count")

    records = []
    last_index, last_timestamp = -1, -math.inf
    for uid, (frame, row) in enumerate(zip(frames, association_rows)):
        _fields(frame, "raw_frame_index timestamp_s frame_id pose_c2w media", "frame")
        index, timestamp, frame_id = (frame[key] for key in ("raw_frame_index", "timestamp_s", "frame_id"))
        _require(type(index) is int and index > last_index and (uid != 0 or index == 0)
                 and frame_id == f"frame_{index:06d}" and type(timestamp) in (int, float)
                 and math.isfinite(timestamp) and timestamp > last_timestamp, "Invalid mapper frame chronology")
        last_index, last_timestamp = index, timestamp

        pose = np.asarray(frame["pose_c2w"], dtype=np.float64)
        _require(pose.shape == (4, 4) and np.isfinite(pose).all()
                 and np.array_equal(pose[3], [0, 0, 0, 1]), f"Invalid homogeneous camera pose: {frame_id}")
        rotation = pose[:3, :3]
        _require(np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=2e-5)
                 and abs(np.linalg.det(rotation) - 1) <= 2e-5, f"Invalid rotation in camera pose: {frame_id}")
        w2c = np.linalg.inv(pose)
        image, depth, paths = _read_media(root, frame["media"], frame_id, (width, height))
        _require(row.split() == [str(timestamp), f"rgb/{frame_id}.png", str(timestamp), f"depth/{frame_id}.png"],
                 f"Incorrect association row: {frame_id}")
        records.append(dict(uid=uid, R=w2c[:3, :3].T, T=w2c[:3, 3], FovY=fovy, FovX=fovx,
                            image=image, image_path=paths["rgb"], image_name=frame_id,
                            width=width, height=height, depth=depth, depth_path=paths["depth"],
                            pose_gt=pose, cx=cx, cy=cy, depth_scale=1000, timestamp=timestamp))

    if camera_info_type is None:
        from scene.dataset_readers import CameraInfo

        camera_info_type = CameraInfo
    return [camera_info_type(**record) for record in records]
