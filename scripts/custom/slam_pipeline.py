#!/usr/bin/env python3

import os
import math
import argparse
import numpy as np
import yaml
import shutil
import subprocess
import matplotlib.pyplot as plt

# evo imports
from rosbags.rosbag1 import Reader
from evo.core import sync
from evo.core.metrics import APE, PoseRelation, Unit
from evo.tools.file_interface import read_bag_trajectory
from evo.tools.plot import PlotCollection, prepare_axis, PlotMode, traj, traj_xyz, traj_rpy, speeds

# ROS & TF
import rosbag
import tf.transformations as tf
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path

def merge_bags(input_bag_dir):
    """Merge multiple rosbag files into one."""
    # copy the rosbag files to the output directory 
    merge_dir = os.path.join(input_bag_dir, "merge")
    os.makedirs(merge_dir, exist_ok=True)
    for bag_file in os.listdir(input_bag_dir):
        if bag_file.endswith(".bag"):
            src_path = os.path.join(input_bag_dir, bag_file)
            dest_path = os.path.join(merge_dir, bag_file)
            # copy the bag file to the merge directory
            shutil.copy2(src_path, dest_path)
            # shutil.move(src_path, dest_path)
            
    cmd = f"rosbag-merge --input_path {merge_dir} --output_path {input_bag_dir} --outbag_name merged_poses --write_bag"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Merged bags into {input_bag_dir}/merged_poses.bag")
    
    # remove the merge directory
    shutil.rmtree(merge_dir)
    print(f"Removed temporary merge directory: {merge_dir}")
    
# -----------------------------------------------------------------------------
def pyify(obj):
    """Recursively turn numpy types into native Python types for YAML."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {pyify(k): pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pyify(v) for v in obj]
    return obj

# -----------------------------------------------------------------------------
def run_evo_traj(bag_path):
    base_dir = os.path.dirname(bag_path)
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # load
    bag = Reader(bag_path); bag.open()
    traj_ref       = read_bag_trajectory(bag, "/groundtruth")
    traj_fastlio2  = read_bag_trajectory(bag, "/Odometry")
    traj_vinfusion = read_bag_trajectory(bag, "/vins_estimator/odometry")
    bag.close()

    # associate
    ref_f, est_f = sync.associate_trajectories(traj_ref, traj_fastlio2,
                                               max_diff=0.001,
                                            #    max_diff=0.02,
                                               first_name="groundtruth",
                                               snd_name="fastlio2")
    ref_v, est_v = sync.associate_trajectories(traj_ref, traj_vinfusion,
                                               max_diff=0.001,
                                            #    max_diff=0.02,
                                               first_name="groundtruth",
                                               snd_name="vinfusion")

    # choose n
    n = min(est_f.num_poses, est_v.num_poses)
    print(f"[evo_traj] using n={n}")

    # align
    r_f, t_f, s_f = est_f.align(ref_f, correct_scale=True,
                                correct_only_scale=False, n=n)
    r_v, t_v, s_v = est_v.align(ref_v, correct_scale=True,
                                correct_only_scale=False, n=n)

    # build Ts
    T_f = np.eye(4); T_f[:3,:3], T_f[:3,3] = r_f, t_f
    T_v = np.eye(4); T_v[:3,:3], T_v[:3,3] = r_v, t_v

    # collect
    results = {}
    for name, (ref_sync, est_sync), (T, s) in [
        ("fastlio2",  (ref_f, est_f),  (T_f, s_f)),
        ("vinfusion", (ref_v, est_v),  (T_v, s_v)),
    ]:
        entry = {
            "infos":      est_sync.get_infos(),
            "checks":     est_sync.check()[1],
            "statistics": est_sync.get_statistics(),
        }
        ape = APE(pose_relation=PoseRelation.full_transformation)
        ape.process_data((ref_sync, est_sync))
        entry["ape"] = ape.get_all_statistics()
        entry["alignment_transform"] = T.tolist()
        entry["scale"]               = float(s)
        results[name] = entry

    clean = pyify(results)
    yaml_path = os.path.join(out_dir, "evaluation_results.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
    print(f"[evo_traj] wrote YAML → {yaml_path}")

    # combined plots
    # (reuse traj_ref, est_f, est_v)
    pc = PlotCollection("All Trajectories Evaluation")

    # 3D
    fig1 = plt.figure(figsize=(8,6))
    ax1  = prepare_axis(fig1, PlotMode.xyz, length_unit=Unit("m"))
    traj(ax1, PlotMode.xyz, traj_ref, style="--", color="black",  label="groundtruth")
    traj(ax1, PlotMode.xyz, est_f,   style="-",  color="red",    label="fastlio2")
    traj(ax1, PlotMode.xyz, est_v,   style="-.", color="blue",   label="vinfusion")
    ax1.set_title("3D Trajectories"); ax1.legend()
    fig1.savefig(os.path.join(out_dir, "trajectory_3D.pdf"))
    pc.add_figure("trajectory_3D", fig1)

    # XYZ vs Time
    fig2, axs2 = plt.subplots(3, sharex="col", figsize=(8,6))
    traj_xyz(axs2, traj_ref, style="--", color="black",  label="groundtruth")
    traj_xyz(axs2, est_f,   style="-",  color="red",    label="fastlio2")
    traj_xyz(axs2, est_v,   style="-.", color="blue",   label="vinfusion")
    axs2[0].set_title("X / Y / Z vs Time")
    axs2[2].legend(loc="upper right")
    fig2.savefig(os.path.join(out_dir, "xyz_vs_time.pdf"))
    pc.add_figure("xyz_vs_time", fig2)

    # RPY vs Time
    fig3, axs3 = plt.subplots(3, sharex="col", figsize=(8,6))
    traj_rpy(axs3, traj_ref, style="--", color="black",  label="groundtruth")
    traj_rpy(axs3, est_f,   style="-",  color="red",    label="fastlio2")
    traj_rpy(axs3, est_v,   style="-.", color="blue",   label="vinfusion")
    axs3[0].set_title("Roll / Pitch / Yaw vs Time")
    axs3[2].legend(loc="upper right")
    fig3.savefig(os.path.join(out_dir, "rpy_vs_time.pdf"))
    pc.add_figure("rpy_vs_time", fig3)

    # Speed
    fig4 = plt.figure(figsize=(8,4))
    ax4  = fig4.gca()
    speeds(ax4, traj_ref, style="--", color="black",  label="groundtruth")
    speeds(ax4, est_f,   style="-",  color="red",    label="fastlio2")
    speeds(ax4, est_v,   style="-.", color="blue",   label="vinfusion")
    ax4.set_title("Speed Profile"); ax4.legend()
    fig4.savefig(os.path.join(out_dir, "speed_profile.pdf"))
    pc.add_figure("speed_profile", fig4)

    print(f"[evo_traj] plots in {out_dir}")
    return yaml_path

# -----------------------------------------------------------------------------
ALGO_TO_TOPIC = {
    "fastlio2":   "/Odometry",
    "vinfusion":  "/vins_estimator/odometry",
    "groundtruth": "/groundtruth",
}

def load_alignments(yaml_path):
    data = yaml.safe_load(open(yaml_path, "r"))
    topic_align = {}
    for algo, entry in data.items():
        if algo not in ALGO_TO_TOPIC:
            continue
        topic = ALGO_TO_TOPIC[algo]
        T = np.array(entry["alignment_transform"], dtype=float)
        topic_align[topic] = {
            "rotation":    T[:3,:3],
            "translation": T[:3, 3],
            "scale":       float(entry.get("scale",1.0))
        }
    return topic_align

def transform_pose(pose, R, tr, s):
    p = np.array([pose.position.x, pose.position.y, pose.position.z])
    p = R.dot(s*p) + tr
    pose.position.x, pose.position.y, pose.position.z = p.tolist()
    M = np.eye(4); M[:3,:3] = R
    q_rot = tf.quaternion_from_matrix(M)
    q_orig = [pose.orientation.x, pose.orientation.y,
              pose.orientation.z, pose.orientation.w]
    q_new  = tf.quaternion_multiply(q_rot, q_orig)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q_new
    return pose

def transform_from_yaml(input_bag, yaml_path, output_bag):
    aligns = load_alignments(yaml_path)
    print(f"[transform] applying to {list(aligns)}")
    with rosbag.Bag(output_bag, 'w') as outbag, rosbag.Bag(input_bag, 'r') as inbag:
        for topic, msg, t in inbag.read_messages():
            if topic in aligns:
                R, tr, s = aligns[topic]["rotation"], aligns[topic]["translation"], aligns[topic]["scale"]
                if isinstance(msg, PoseStamped):
                    msg.pose = transform_pose(msg.pose, R, tr, s)
                elif isinstance(msg, PoseWithCovarianceStamped):
                    msg.pose.pose = transform_pose(msg.pose.pose, R, tr, s)
                elif isinstance(msg, Odometry):
                    msg.pose.pose = transform_pose(msg.pose.pose, R, tr, s)
                else:
                    # fallback
                    if hasattr(msg, "pose"):
                        inner = msg.pose
                        if hasattr(inner, "position"):
                            msg.pose = transform_pose(inner, R, tr, s)
                        elif hasattr(inner, "pose"):
                            msg.pose.pose = transform_pose(inner.pose, R, tr, s)
            outbag.write(topic, msg, t)
    print(f"[transform] wrote {output_bag}")
    return output_bag

def modify_frame_id(input_bag, output_bag, new_frame):
    from rosbag.bag import Bag
    def _recurse(m):
        if hasattr(m, "header") and hasattr(m.header, "frame_id"):
            m.header.frame_id = new_frame
        for name in dir(m):
            if name.startswith("_"):
                continue
            v = getattr(m, name)
            if isinstance(v, list):
                for item in v:
                    _recurse(item)
            elif hasattr(v, "header"):
                _recurse(v)

    with Bag(output_bag, 'w') as outbag, Bag(input_bag, 'r') as inbag:
        for topic, msg, t in inbag.read_messages():
            _recurse(msg)
            outbag.write(topic, msg, t)
    print(f"[modify_frame_id] wrote {output_bag}")
    return output_bag

def append_pose(path_msg: Path, ps: PoseStamped):
    if not path_msg.poses:
        path_msg.header.frame_id = ps.header.frame_id
    path_msg.header.stamp = ps.header.stamp
    path_msg.poses.append(ps)

def pose_to_path(input_bag, output_bag):
    paths = {}
    with rosbag.Bag(output_bag, 'w') as outbag, rosbag.Bag(input_bag, 'r') as inbag:
        for topic, msg, t in inbag.read_messages():
            outbag.write(topic, msg, t)
            mtype = getattr(msg, "_type", "")
            if mtype == "geometry_msgs/PoseStamped":
                ps = msg
            elif mtype == "nav_msgs/Odometry":
                ps = PoseStamped(header=msg.header, pose=msg.pose.pose)
            else:
                continue
            pt = topic.rstrip("/") + "/path"
            if pt not in paths:
                paths[pt] = Path()
            append_pose(paths[pt], ps)
            outbag.write(pt, paths[pt], t)
    print(f"[pose_to_path] wrote {output_bag}")
    return output_bag

# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Full SLAM‐evaluation & pipeline")
    p.add_argument("--bag",             required=True, help="Original rosbag")
    p.add_argument("--frame-id",        default="odom", help="New header frame_id")
    p.add_argument("--no-intermediate", action="store_true",
                   help="Delete intermediate bag files after run")
    args = p.parse_args()

    bag_path = args.bag
    base_dir = os.path.dirname(bag_path)

    # Merge bags if needed
    # if os.path.isdir(bag_path):
    if os.path.exists(bag_path):
        print(f"Bag file is a directory, merging bags...")
        merge_bags(base_dir)
        bag_path = os.path.join(base_dir, "merged_poses.bag")
        print(f"Merged bag file: {bag_path}")
    else:
        print(f"Bag file {bag_path} does not exist.")
        return
    
    # 1) EVO → YAML + PDFs
    yaml_path = run_evo_traj(bag_path)

    # 2) apply transforms
    transformed_bag   = os.path.join(base_dir, "transformed.bag")
    fixedframe_bag    = os.path.join(base_dir, "tfed_fixedframe.bag")
    final_path_bag    = os.path.join(base_dir, "merged_poses_pathes.bag")

    transform_from_yaml(bag_path, yaml_path, transformed_bag)
    modify_frame_id(transformed_bag, fixedframe_bag, args.frame_id)
    pose_to_path(fixedframe_bag, final_path_bag)

    if args.no_intermediate:
        for f in (transformed_bag, fixedframe_bag):
            try:
                os.remove(f)
                print(f"[cleanup] removed {f}")
            except FileNotFoundError:
                pass

    print("\n✅ Pipeline complete!")
    print("  evaluation_results.yaml:", yaml_path)
    print("  final bag with paths:    ", final_path_bag)

if __name__ == "__main__":
    main()
