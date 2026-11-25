"""CPU tests for the fixed-calibration mapper input boundary."""

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
import pytest

from fixed_rgbd import load_fixed_rgbd


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


@pytest.fixture
def camera_info_type():
    # Import the checked-out record declaration without initializing the CUDA
    # globals imported by scene.dataset_readers.
    source = Path(__file__).resolve().parents[2] / "scene/dataset_readers.py"
    declaration = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "CameraInfo"
    )
    namespace = {"NamedTuple": NamedTuple, "np": np}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["CameraInfo"]


def write_manifest(root, payload):
    body = {key: value for key, value in payload.items() if key != "input_sha256"}
    manifest = root / "input.json"
    manifest.write_text(json.dumps({**body, "input_sha256": canonical(body)}))
    return manifest, digest(manifest.read_bytes())


@pytest.fixture
def packet(tmp_path):
    frames = []
    for index in (0, 7):
        frame_id = f"frame_{index:06d}"
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [1 + index, 2, 3]
        media = {}
        for kind, array in {
            "rgb": np.array([[[1, 20, 250], [0, 0, 0]], [[4, 6, 8], [255, 2, 8]]], np.uint8),
            "depth": np.array([[1000, 0], [1800, 3791]], np.uint16),
            "masks": np.array([[255, 0], [255, 255]], np.uint8),
        }.items():
            relative = f"{kind}/{frame_id}.png"
            path = tmp_path / relative
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(array).save(path)
            media[kind] = {"path": relative, "sha256": digest(path.read_bytes())}
        frames.append(dict(raw_frame_index=index, timestamp_s=1200.25 + index * 0.2,
                           frame_id=frame_id, pose_c2w=pose.tolist(), media=media))
    calibration = {"width": 2, "height": 2, "K": [[5, 0, 0.8], [0, 6, 1.2], [0, 0, 1]],
                   "frame_id": frames[0]["frame_id"]}
    associations = "".join(
        f"{f['timestamp_s']} {f['media']['rgb']['path']} "
        f"{f['timestamp_s']} {f['media']['depth']['path']}\n" for f in frames
    ).encode()
    (tmp_path / "associations.txt").write_bytes(associations)
    payload = dict(schema="sceneweave-cartgs-fixed-calibration-input-v1", role="mapper_input",
                   scene_id="fixture", source_prefix_sha256="1" * 64,
                   transform_version="registered-fixed-k00-v1",
                   pose_axes="opencv_camera_to_zup_world", pose_protocol="frame0_relative_arkit_vio",
                   depth_units_per_meter=1000, calibration=calibration,
                   fixed_calibration_sha256=canonical(calibration),
                   associations_sha256=digest(associations), frames=frames)
    return tmp_path, payload


def test_preserves_media_geometry_and_causal_poses(packet, camera_info_type):
    root, payload = packet
    manifest, pin = write_manifest(root, payload)
    cameras = load_fixed_rgbd(manifest, expected_manifest_sha256=pin, camera_info_type=camera_info_type)
    assert len(cameras) == 2 and type(cameras[0]) is camera_info_type
    for uid, (camera, frame) in enumerate(zip(cameras, payload["frames"])):
        assert (camera.uid, camera.image_name, camera.timestamp) == (uid, frame["frame_id"], frame["timestamp_s"])
        assert (camera.width, camera.height, camera.cx, camera.cy, camera.depth_scale) == (2, 2, 0.8, 1.2, 1000)
        np.testing.assert_allclose([camera.FovX, camera.FovY], 2 * np.arctan([1 / 5, 1 / 6]))
        pose = np.asarray(frame["pose_c2w"])
        np.testing.assert_array_equal(camera.pose_gt, pose)
        w2c = np.linalg.inv(pose)
        np.testing.assert_allclose(camera.R.T, w2c[:3, :3])
        np.testing.assert_allclose(camera.T, w2c[:3, 3])
        with Image.open(camera.image_path) as rgb:
            np.testing.assert_array_equal(np.asarray(camera.image), np.asarray(rgb))
        assert camera.depth.mode == "F"
        np.testing.assert_array_equal(np.asarray(camera.depth), np.array([[1, 0], [1.8, 3.791]], np.float32))


@pytest.mark.parametrize("mutation,match", [
    ("evaluator", "fields"),
    ("traversal", "media path"),
    ("symlink", "media path"),
    ("chronology", "chronology"),
    ("reflection", "pose"),
    ("calibration", "calibration"),
    ("associations", "association"),
    ("mask", "mask"),
])
def test_rejects_invalid_resigned_packet(packet, camera_info_type, mutation, match):
    root, payload = packet
    frame = payload["frames"][1]
    if mutation == "evaluator":
        frame["ground_truth"] = "evaluator/mesh.ply"
    elif mutation == "traversal":
        frame["media"]["depth"]["path"] = "../evaluator.png"
    elif mutation == "symlink":
        path = root / frame["media"]["depth"]["path"]
        evaluator = root / "evaluator.png"
        path.rename(evaluator)
        path.symlink_to(evaluator)
    elif mutation == "chronology":
        frame["timestamp_s"] = payload["frames"][0]["timestamp_s"]
    elif mutation == "reflection":
        frame["pose_c2w"][2][2] = -1
    elif mutation == "calibration":
        payload["calibration"]["K"][0][1] = 0.01
        payload["fixed_calibration_sha256"] = canonical(payload["calibration"])
    elif mutation == "associations":
        associations = root / "associations.txt"
        associations.write_text(associations.read_text().replace("1200.25", "1200.0"))
        payload["associations_sha256"] = digest(associations.read_bytes())
    elif mutation == "mask":
        entry = frame["media"]["masks"]
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / entry["path"])
        entry["sha256"] = digest((root / entry["path"]).read_bytes())
    manifest, pin = write_manifest(root, payload)
    with pytest.raises(ValueError, match=match):
        load_fixed_rgbd(manifest, expected_manifest_sha256=pin, camera_info_type=camera_info_type)


def test_checks_external_pin_and_media_integrity(packet, camera_info_type):
    root, payload = packet
    manifest, pin = write_manifest(root, payload)
    with pytest.raises(ValueError, match="manifest digest"):
        load_fixed_rgbd(manifest, expected_manifest_sha256="0" * 64, camera_info_type=camera_info_type)
    media = root / payload["frames"][1]["media"]["rgb"]["path"]
    media.write_bytes(media.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="media digest"):
        load_fixed_rgbd(manifest, expected_manifest_sha256=pin, camera_info_type=camera_info_type)


@pytest.mark.parametrize("name", ["office_m40", "kitchen_m40"])
def test_real_staged_packet(camera_info_type, name):
    packet_root = os.environ.get("RTG_FIXED_RGBD_PACKET_ROOT")
    if not packet_root:
        pytest.skip("set RTG_FIXED_RGBD_PACKET_ROOT to the staged office/kitchen directory")
    root = Path(packet_root) / name
    report = json.loads((root / "report.json").read_text())
    manifest = root / "input/input.json"
    payload = json.loads(manifest.read_text())
    assert payload["source_prefix_sha256"] == report["source_prefix_sha256"]
    assert payload["fixed_calibration_sha256"] == report["fixed_calibration_sha256"]
    cameras = load_fixed_rgbd(manifest, expected_manifest_sha256=report["input_manifest_sha256"],
                             camera_info_type=camera_info_type)
    assert len(cameras) == len(payload["frames"])
    assert not {cam.image_name for cam in cameras}.intersection(report["holdout_frame_ids"])
    k = np.asarray(payload["calibration"]["K"])
    for camera, frame in zip(cameras, payload["frames"]):
        assert (camera.width, camera.height) == (256, 192)
        assert camera.timestamp == frame["timestamp_s"]
        np.testing.assert_array_equal(camera.pose_gt, frame["pose_c2w"])
        w2c = np.linalg.inv(frame["pose_c2w"])
        np.testing.assert_allclose(camera.R.T, w2c[:3, :3])
        np.testing.assert_allclose(camera.T, w2c[:3, 3])
        np.testing.assert_allclose([camera.width / (2 * np.tan(camera.FovX / 2)),
                                    camera.height / (2 * np.tan(camera.FovY / 2)),
                                    camera.cx, camera.cy], [k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        with Image.open(camera.image_path) as rgb, Image.open(camera.depth_path) as depth:
            np.testing.assert_array_equal(np.asarray(camera.image), np.asarray(rgb))
            np.testing.assert_array_equal(np.asarray(camera.depth), np.asarray(depth, dtype=np.float32) / 1000)
