#!/usr/bin/env python3
import rosbag
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# —————— EDIT THESE ———————
input_bag  = '/media/puonn/zw/puonn/fusion_portable/v2/processed_data/ugv/campus/00/tfed_fixedframe.bag'
output_bag = '/media/puonn/zw/puonn/fusion_portable/v2/processed_data/ugv/campus/00/tfed_fxed_path.bag'
# ——————————————————————

# container for growing Path messages
paths = {}

def append_pose(path: Path, ps: PoseStamped):
    """Append a PoseStamped to a Path, updating header/frame."""
    if not path.poses:
        path.header.frame_id = ps.header.frame_id
    path.header.stamp = ps.header.stamp
    path.poses.append(ps)

with rosbag.Bag(output_bag, 'w') as outbag, rosbag.Bag(input_bag, 'r') as inbag:
    for topic, msg, t in inbag.read_messages():
        # 1) copy original msg
        outbag.write(topic, msg, t)

        # 2) detect type by msg._type
        mtype = getattr(msg, '_type', '')
        if mtype == 'geometry_msgs/PoseStamped':
            # directly use it
            ps = msg

        elif mtype == 'nav_msgs/Odometry':
            # extract inner pose
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose   = msg.pose.pose

        else:
            # not a trajectory message we care about
            continue

        # 3) build or append to Path under "<topic>/path"
        path_topic = topic.rstrip('/') + '/path'
        if path_topic not in paths:
            paths[path_topic] = Path()
        append_pose(paths[path_topic], ps)
        outbag.write(path_topic, paths[path_topic], t)

print("✅ Done!  Now run:")
print(f"    rosbag info {output_bag}")
print("and you should see `/groundtruth/path` (and the other `/…/path` topics).")
