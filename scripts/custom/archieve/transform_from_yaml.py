#!/usr/bin/env python3
import argparse
import yaml
import rosbag
import numpy as np
import tf.transformations as tf

from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
)
from nav_msgs.msg import Odometry

# === 1) Map your algorithm names (keys in the YAML) to the ROS topics ===
ALGO_TO_TOPIC = {
    "fastlio2":  "/Odometry",
    "vinfusion": "/vins_estimator/odometry",
    # "r3live":    "/",
    "groundtruth": "/groundtruth",
}

def load_alignments(yaml_path):
    """
    Load alignment transforms and scales from a YAML file.
    Returns dict: topic -> {"rotation": R, "translation": t, "scale": s}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # if nested under 'results', uncomment:
    # data = data.get("results", data)

    topic_align = {}
    for algo_name, entry in data.items():
        if algo_name not in ALGO_TO_TOPIC:
            continue
        topic = ALGO_TO_TOPIC[algo_name]
        T_list = entry.get("alignment_transform")
        s      = entry.get("scale", 1.0)
        if T_list is None:
            raise KeyError(f"No alignment_transform for '{algo_name}' in YAML")
        T = np.array(T_list, dtype=float)
        R = T[:3, :3]
        t = T[:3,  3]
        topic_align[topic] = {"rotation": R, "translation": t, "scale": float(s)}
    return topic_align

def transform_pose(pose, rotation, translation, scale):
    """
    Apply: scaled → rotated → translated to pose.position,
    and rotate pose.orientation accordingly.
    """
    # position
    p = np.array([pose.position.x, pose.position.y, pose.position.z])
    p = rotation.dot(scale * p) + translation
    pose.position.x, pose.position.y, pose.position.z = p.tolist()

    # orientation
    T4 = np.eye(4)
    T4[:3, :3] = rotation
    q_rot = tf.quaternion_from_matrix(T4)
    q_orig = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]
    q_new = tf.quaternion_multiply(q_rot, q_orig)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q_new

    return pose

def main():
    p = argparse.ArgumentParser(
        description="Apply saved alignment+scale from YAML to a rosbag"
    )
    p.add_argument("--input-bag",  required=True, help="Path to input rosbag")
    p.add_argument("--yaml",       required=True, help="Path to evaluation_results.yaml")
    p.add_argument("--output-bag", required=True, help="Path to output (transformed) rosbag")
    args = p.parse_args()

    alignments = load_alignments(args.yaml)
    if not alignments:
        raise RuntimeError("No alignments found in YAML for known algorithms")

    print("Applying alignments to topics:")
    for topic in alignments:
        print("  ", topic)

    with rosbag.Bag(args.output_bag, "w") as outbag, \
         rosbag.Bag(args.input_bag,  "r") as inbag:

        for topic, msg, t in inbag.read_messages():
            if topic in alignments:
                a = alignments[topic]
                R, tr, s = a["rotation"], a["translation"], a["scale"]

                # PoseStamped
                if isinstance(msg, PoseStamped):
                    msg.pose = transform_pose(msg.pose, R, tr, s)

                # PoseWithCovarianceStamped
                elif isinstance(msg, PoseWithCovarianceStamped):
                    msg.pose.pose = transform_pose(msg.pose.pose, R, tr, s)

                # Odometry
                elif isinstance(msg, Odometry):
                    msg.pose.pose = transform_pose(msg.pose.pose, R, tr, s)

                # fallback for any other msg that has .pose or .pose.pose
                else:
                    # try PoseWithCovariance (e.g. msg.pose.pose)
                    if hasattr(msg, "pose"):
                        inner = msg.pose
                        if hasattr(inner, "position"):
                            # msg.pose is actually a Pose
                            msg.pose = transform_pose(inner, R, tr, s)
                        elif hasattr(inner, "pose"):
                            # msg.pose.pose is the Pose
                            msg.pose.pose = transform_pose(inner.pose, R, tr, s)

            outbag.write(topic, msg, t)

    print(f"Done. Transformed bag saved to {args.output_bag}")

if __name__ == "__main__":
    main()