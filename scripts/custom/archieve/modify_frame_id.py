#!/usr/bin/env python

import rosbag
import rospy
from rosbag.bag import Bag

# Input and output bag paths
input_bag = '/media/puonn/zw/puonn/fusion_portable/v2/processed_data/ugv/campus/00/transformed.bag'   # Replace with your input bag file
output_bag = '/media/puonn/zw/puonn/fusion_portable/v2/processed_data/ugv/campus/00/tfed_fixedframe.bag'  # Name for the output bag with modified frame_id


def modify_frame_id():
    with Bag(output_bag, 'w') as outbag:
        # Read the input bag
        with Bag(input_bag, 'r') as inbag:
            for topic, msg, t in inbag.read_messages():
                # Recursive function to modify all 'frame_id' fields in message objects
                def modify_msg_frame_id(msg):
                    if hasattr(msg, 'header') and hasattr(msg.header, 'frame_id'):
                        msg.header.frame_id = "odom"  # Change to "world" or any other desired frame_id

                    # Check nested messages (e.g., messages with arrays or sub-messages)
                    for slot_name in dir(msg):
                        if not slot_name.startswith('_'):
                            sub_msg = getattr(msg, slot_name)
                            if isinstance(sub_msg, list):
                                for item in sub_msg:
                                    modify_msg_frame_id(item)
                            elif hasattr(sub_msg, 'header'):
                                modify_msg_frame_id(sub_msg)
                    
                modify_msg_frame_id(msg)
                
                # Write the modified message to the new bag
                outbag.write(topic, msg, t)
                
    print("Modified ROS bag written to {}".format(output_bag))

if __name__ == '__main__':
    modify_frame_id()
