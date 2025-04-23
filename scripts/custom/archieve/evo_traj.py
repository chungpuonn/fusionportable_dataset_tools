#!/usr/bin/env python3

import os
import math
import numpy as np
import yaml
import matplotlib.pyplot as plt
from rosbags.rosbag1 import Reader

from evo.core import sync
from evo.core.metrics import APE, PoseRelation, Unit
from evo.tools.file_interface import read_bag_trajectory
from evo.tools.plot import (
    PlotCollection,
    prepare_axis,
    PlotMode,
    traj,
    traj_xyz,
    traj_rpy,
    speeds
)

def pyify(obj):
    """
    Recursively convert numpy types into native Python types so that
    yaml.safe_dump can serialize them without errors.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {pyify(k): pyify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [pyify(v) for v in obj]
    return obj

def main():
    # — Paths & output directory —
    bag_path = "/media/puonn/zw/puonn/fusion_portable/v2/processed_data/ugv/campus/00/merged.bag"
    out_dir  = os.path.join(os.path.dirname(bag_path), "results")
    os.makedirs(out_dir, exist_ok=True)

    # — Load trajectories —
    bag = Reader(bag_path)
    bag.open()
    traj_ref       = read_bag_trajectory(bag, "/groundtruth")
    traj_fastlio2  = read_bag_trajectory(bag, "/Odometry")
    traj_vinfusion = read_bag_trajectory(bag, "/vins_estimator/odometry")
    bag.close()

    # — Associate & Align —
    ref_f, est_f = sync.associate_trajectories(
        traj_ref, traj_fastlio2, max_diff=0.02,
        first_name="groundtruth", snd_name="fastlio2"
    )
    ref_v, est_v = sync.associate_trajectories(
        traj_ref, traj_vinfusion, max_diff=0.02,
        first_name="groundtruth", snd_name="vinfusion"
    )
    
    # — compute a single n based on the smallest trajectory length ÷ 4 —
    n = min(est_f.num_poses, est_v.num_poses) 
    print(f"Using n={n} for both alignments (of smallest traj)") 

    # Umeyama alignment parameters
    r_f, t_f, s_f = est_f.align(ref_f,
                                correct_scale=True,
                                correct_only_scale=False,
                                n=n)
    r_v, t_v, s_v = est_v.align(ref_v,
                                correct_scale=True,
                                correct_only_scale=False,
                                n=n)

    # Build homogeneous transforms
    T_fastlio2 = np.eye(4)
    T_fastlio2[:3, :3] = r_f
    T_fastlio2[:3,  3] = t_f

    T_vinfusion = np.eye(4)
    T_vinfusion[:3, :3] = r_v
    T_vinfusion[:3,  3] = t_v

    # — Collect evaluation results —
    results = {}
    for name, (r_sync, e_sync), (T, s) in [
        ("fastlio2",  (ref_f, est_f),   (T_fastlio2, s_f)),
        ("vinfusion", (ref_v, est_v),   (T_vinfusion, s_v)),
    ]:
        entry = {}
        entry["infos"]      = e_sync.get_infos()
        entry["checks"]     = e_sync.check()[1]
        entry["statistics"] = e_sync.get_statistics()

        # APE metrics
        ape = APE(pose_relation=PoseRelation.full_transformation)
        ape.process_data((r_sync, e_sync))
        entry["ape"] = ape.get_all_statistics()

        # Convert the transform to nested lists immediately
        entry["alignment_transform"] = T.tolist()
        entry["scale"]               = float(s)        # <— save the scale!
        
        results[name] = entry

    # Convert any remaining numpy objects recursively
    clean_results = pyify(results)

    # Write to YAML
    yaml_path = os.path.join(out_dir, "evaluation_results.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(clean_results, f, sort_keys=False)

    # — Generate & save combined plots —
    pc = PlotCollection("All Trajectories Evaluation")

    # 3D Trajectories
    fig1 = plt.figure(figsize=(8,6))
    ax1  = prepare_axis(fig1, PlotMode.xyz, length_unit=Unit("m"))
    traj(ax1, PlotMode.xyz, ref_f, style="--", color="black", label="groundtruth")
    traj(ax1, PlotMode.xyz, est_f, style="-",  color="red",   label="fastlio2")
    traj(ax1, PlotMode.xyz, est_v, style="-.", color="blue",  label="vinfusion")
    ax1.set_title("3D Trajectories")
    ax1.legend()
    fig1.savefig(os.path.join(out_dir, "trajectory_3D.pdf"))
    pc.add_figure("trajectory_3D", fig1)

    # X/Y/Z vs Time
    fig2, axs2 = plt.subplots(3, sharex="col", figsize=(8,6))
    traj_xyz(axs2, ref_f, style="--", color="black",  label="groundtruth")
    traj_xyz(axs2, est_f, style="-",  color="red",    label="fastlio2")
    traj_xyz(axs2, est_v, style="-.", color="blue",   label="vinfusion")
    axs2[0].set_title("X / Y / Z vs Time")
    axs2[2].legend(loc="upper right")
    fig2.savefig(os.path.join(out_dir, "xyz_vs_time.pdf"))
    pc.add_figure("xyz_vs_time", fig2)

    # Roll/Pitch/Yaw vs Time
    fig3, axs3 = plt.subplots(3, sharex="col", figsize=(8,6))
    traj_rpy(axs3, ref_f, style="--", color="black",  label="groundtruth")
    traj_rpy(axs3, est_f, style="-",  color="red",    label="fastlio2")
    traj_rpy(axs3, est_v, style="-.", color="blue",   label="vinfusion")
    axs3[0].set_title("Roll / Pitch / Yaw vs Time")
    axs3[2].legend(loc="upper right")
    fig3.savefig(os.path.join(out_dir, "rpy_vs_time.pdf"))
    pc.add_figure("rpy_vs_time", fig3)

    # Speed Profile
    fig4 = plt.figure(figsize=(8,4))
    ax4  = fig4.gca()
    speeds(ax4, ref_f, style="--", color="black",  label="groundtruth")
    speeds(ax4, est_f, style="-",  color="red",    label="fastlio2")
    speeds(ax4, est_v, style="-.", color="blue",   label="vinfusion")
    ax4.set_title("Speed Profile")
    ax4.legend()
    fig4.savefig(os.path.join(out_dir, "speed_profile.pdf"))
    pc.add_figure("speed_profile", fig4)

    # Optionally display them
    pc.show()

if __name__ == "__main__":
    main()
