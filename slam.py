import os
from argparse import ArgumentParser

from utils.config_utils import read_config
parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--config", type=str, default="configs/replica/office0.yaml")
args = parser.parse_args()
config_path = args.config
args = read_config(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.device_list)
import torch
import json
import time
from utils.camera_utils import loadCam
from arguments import DatasetParams, MapParams, OptimizationParams
from scene import Dataset
from SLAM.multiprocess.mapper import Mapping
from SLAM.multiprocess.tracker import Tracker, convert_poses
from SLAM.utils import *
from SLAM.eval import eval_frame
from utils.general_utils import safe_state
from utils.monitor import Recorder

torch.set_printoptions(4, sci_mode=False)


def main():
    # set visible devices
    time_recorder = Recorder(args.device_list[0])
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser, sentinel=True)
    map_params = MapParams(parser)

    safe_state(args.quiet)
    optimization_params = optimization_params.extract(args)
    dataset_params = dataset_params.extract(args)
    map_params = map_params.extract(args)

    # Initialize dataset
    dataset = Dataset(
        dataset_params,
        shuffle=False,
        resolution_scales=dataset_params.resolution_scales,
    )

    record_mem = args.record_mem

    gaussian_map = Mapping(args, time_recorder)
    gaussian_map.create_workspace()
    gaussian_tracker = Tracker(args)
    # save config file
    prepare_cfg(args)
    # set time log
    tracker_time_sum = 0
    mapper_time_sum = 0

    # Opt-in synchronized qualification; ordinary algorithm path remains unchanged.
    qualification = os.environ.get('RTG_QUALIFICATION_JSON')
    samples = []
    def clock():
        if qualification:
            torch.cuda.synchronize()
        return time.perf_counter()
    loop_start = clock()
    if qualification:
        torch.cuda.reset_peak_memory_stats()
    # start SLAM
    for frame_id, frame_info in enumerate(dataset.scene_info.train_cameras):
        curr_frame = loadCam(
            dataset_params, frame_id, frame_info, dataset_params.resolution_scales[0]
        )

        print("\n========== curr frame is: %d ==========\n" % frame_id)
        move_to_gpu(curr_frame)
        start_time = clock()
        # tracker process
        frame_map = gaussian_tracker.map_preprocess(curr_frame, frame_id)
        tracking_success = gaussian_tracker.tracking(curr_frame, frame_map)
        tracker_time = clock()
        tracker_consume_time = tracker_time - start_time
        time_recorder.update_mean("tracking", tracker_consume_time, 1)

        tracker_time_sum += tracker_consume_time
        print(f"[LOG] tracker cost time: {tracker_time - start_time}")

        mapper_start_time = clock()

        new_poses = gaussian_tracker.get_new_poses()
        gaussian_map.update_poses(new_poses)
        # mapper process
        gaussian_map.mapping(curr_frame, frame_map, frame_id, optimization_params)

        gaussian_map.get_render_output(curr_frame)
        gaussian_tracker.update_last_status(
            curr_frame,
            gaussian_map.model_map["render_depth"],
            gaussian_map.frame_map["depth_map"],
            gaussian_map.model_map["render_normal"],
            gaussian_map.frame_map["normal_map_w"],
        )
        mapper_time = clock()
        mapper_consume_time = mapper_time - mapper_start_time
        time_recorder.update_mean("mapping", mapper_consume_time, 1)

        mapper_time_sum += mapper_consume_time
        print(f"[LOG] mapper cost time: {mapper_time - tracker_time}")
        if record_mem:
            time_recorder.watch_gpu()
        quality = None
        # report eval loss
        if ((gaussian_map.time + 1) % gaussian_map.save_step == 0) or (
            gaussian_map.time == 0
        ):
            quality = eval_frame(
                gaussian_map,
                curr_frame,
                os.path.join(gaussian_map.save_path, "eval_render"),
                min_depth=gaussian_map.min_depth,
                max_depth=gaussian_map.max_depth,
                save_picture=True,
                run_pcd=False
            )
            gaussian_map.save_model(save_data=True)

        gaussian_map.time += 1
        move_to_cpu(curr_frame)
        torch.cuda.empty_cache()
        if qualification:
            orb_ok = None
            if gaussian_tracker.orb_backend is not None:
                import orbslam2
                orb_ok = gaussian_tracker.orb_backend.get_tracking_state() == orbslam2.TrackingState.OK
            samples.append(dict(frame=frame_id, timestamp=float(curr_frame.timestamp),
                tracking_s=tracker_consume_time, mapping_s=mapper_consume_time,
                elapsed_s=clock()-loop_start, icp_success=bool(tracking_success), orb_tracking_ok=None if orb_ok is None else bool(orb_ok),
                observed_view_quality=quality))
    online_s = clock()-loop_start
    print("\n========== main loop finish ==========\n")
    print(
        "[LOG] stable num: {:d}, unstable num: {:d}".format(
            gaussian_map.get_stable_num, gaussian_map.get_unstable_num
        )
    )
    print("[LOG] processed frame: ", gaussian_map.optimize_frames_ids)
    print("[LOG] keyframes: ", gaussian_map.keyframe_ids)
    print("[LOG] mean tracker process time: ", tracker_time_sum / (frame_id + 1))
    print("[LOG] mean mapper process time: ", mapper_time_sum / (frame_id + 1))
    
    new_poses = gaussian_tracker.get_new_poses()
    gaussian_map.update_poses(new_poses)
    gaussian_map.global_optimization(optimization_params, is_end=True)
    finalized_s = clock()-loop_start
    eval_frame(
        gaussian_map,
        gaussian_map.keyframe_list[-1],
        os.path.join(gaussian_map.save_path, "eval_render"),
        min_depth=gaussian_map.min_depth,
        max_depth=gaussian_map.max_depth,
        save_picture=True,
        run_pcd=False
    )
    
    gaussian_map.save_model(save_data=True)
    if qualification:
        orb_stamps = None
        if gaussian_tracker.orb_backend is not None:
            _, orb_stamps = convert_poses(gaussian_tracker.orb_backend.get_trajectory_points())
            orb_stamps = [float(v) for v in orb_stamps]
        # Evaluate a fixed set of withheld RGB-D views for mapper parity only.
        heldout = []
        if args.use_gt_pose:
            for i, info in enumerate(dataset.scene_info.test_cameras):
                cam = loadCam(dataset_params, i, info, dataset_params.resolution_scales[0])
                cam.updatePose(info.pose_gt)
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                gaussian_map.time = i
                metrics = eval_frame(gaussian_map, cam, os.path.join(args.save_path, 'heldout'),
                    min_depth=gaussian_map.min_depth, max_depth=gaussian_map.max_depth,
                    save_picture=False, run_pcd=False)
                heldout.append(dict(timestamp=float(info.timestamp), **metrics))
            gaussian_map.time = len(samples)
        # Capture final optimization completion separately from offline ATE/plot/export work.
        result = dict(backend='rocm' if torch.version.hip else 'cuda',
                      device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                      hip=torch.version.hip, cuda=torch.version.cuda,
                      orb_active=gaussian_tracker.orb_backend is not None,
                      orb_timestamps=orb_stamps, use_gt_pose=bool(args.use_gt_pose),
                      frames=samples, heldout=heldout, online_s=online_s, finalized_s=finalized_s,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with open(qualification, 'w') as fp:
            json.dump(result, fp, indent=2, allow_nan=False)
    gaussian_tracker.save_traj(args.save_path)
    time_recorder.cal_fps()
    time_recorder.save(args.save_path)
    gaussian_map.time += 1
    
    if args.pcd_densify:    
        densify_pcd = gaussian_map.stable_pointcloud.densify(1, 30, 5)
        o3d.io.write_point_cloud(
            os.path.join(args.save_path, "save_model", "pcd_densify.ply"), densify_pcd
        )


if __name__ == "__main__":
    main()
