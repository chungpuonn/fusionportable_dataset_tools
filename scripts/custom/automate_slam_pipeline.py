import os
import subprocess
import time
import argparse
import signal
import time
import glob
import shutil
import sys
import rosbag
import rospy
import evo
import re
from pathlib import Path
from typing import Tuple
import sys

# =================== Configuration ===================
# Path to your input dataset rosbag file
root_dir = "/media/puonn/zw/puonn/fusion_portable/v2"
rosbag_dir = "/sensor_data/vehicle_campus00.bag"
rosbag_full_path = root_dir + rosbag_dir
# =======================================================

def bag_to_hierarchy(bag_path: str, root_dir: str = None) -> str:
    """
    Turn a bag filename like '/some/where/ugv_campus00.bag' into
      'ugv/campus/00' (or 'root_dir/ugv/campus/00' if you pass root_dir).

    Raises ValueError if the filename doesn't match <platform>_<env><idx>.bag.
    """
    p = Path(bag_path)
    name = p.stem  # e.g. 'ugv_campus00'
    
    # Pattern: <platform>_<env><idx>
    m = re.match(r"(?P<platform>[^_]+)_(?P<env>[A-Za-z]+)(?P<idx>\d+)$", name)
    if not m:
        raise ValueError(f"Filename {p.name!r} does not match <platform>_<env><idx>.bag")
    
    platform = m.group("platform")
    env      = m.group("env")
    idx      = m.group("idx")
    
     # build the relative hierarchy path
    parts = [platform, env, idx]
    hierarchy_path = os.path.join(*parts)
    
    # if a root_dir was given, prepend it
    if root_dir:
        hierarchy_path = os.path.join(root_dir, hierarchy_path)
    return hierarchy_path, platform, env
    
# ========================================================= 
hierarchy_path, platform_type, scene_type = bag_to_hierarchy(rosbag_full_path)
abs_hierarchy_path = os.path.join(root_dir + '/processed_data/' + hierarchy_path)

# Output directory to store recorded bags, trajectories, evaluation results, and plots
output_dir = "/media/puonn/zw/puonn/fusion_portable/v2/processed_data"

# SLAM algorithms configuration: launch file and ROS output topic for each
slam_algorithms = {
    "fast_lio": {
        "launch_file": "mapping_ouster64.launch",
        "output_topic": "/Odometry"
    },
    "vins": {
        "launch_file": "vins_rviz.launch",
        "config_file": {"ugv": "ramlab_dataset_20230426Calib_ugv/ramlab_stereo_imu_config.yaml",
                        "vehicle": "ramlab_dataset_20230618Calib_vehicle/ramlab_stereo_imu_config.yaml",
                        "handheld": "ramlab_dataset_20230426Calib_handheld/ramlab_stereo_imu_config.yaml",
                        "legged": "ramlab_dataset_20230912Calib_legged/ramlab_stereo_imu_config.yaml"},
        "output_topic": "/vins_estimator/odometry"
    },
    # "r3live": {
    #     "launch_file": "r3live_bag_fusionportable.launch",
    #     "output_topic": "/aft_mapped_to_init"
    # }
}
# ======================================================

def run_command(command):
    """Run a shell command and return its process handle."""
    print(f"Running command: {command}")
    return subprocess.Popen(command, shell=True)

def launch_slam(package_name, details):
    """Launch a SLAM algorithm using its ROS launch file."""
    if package_name == "vins":
        # Check if the config file exists
        config_file = details["config_file"].get(platform_type)
        if not config_file:
            raise ValueError(f"Configuration file for platform '{platform_type}' not found.")
        # Construct the command to launch VINS with the specific config file
        cmd = f"roslaunch {package_name} {details['launch_file']} config_path:=$(rospack find vins)/../config/{config_file}"
    else:
        cmd = f"roslaunch {package_name} {details['launch_file']} "
    return run_command(cmd)

def record_topic(output_path, topic):
    # 1. Enable sim time globally
    subprocess.run("rosparam set /use_sim_time true", shell=True, check=True)
    
    # 2. Start recording both the clock and the SLAM topic
    """Record a specific ROS topic into a rosbag file."""
    cmd = f"rosbag record -O {output_path} /clock {topic}"
    # Create a new process group for the recording process
    return subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

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

def evaluate_trajectory(algorithm_name, output_bag, output_topic, groundtruth_bag):
    """
    Evaluate the SLAM trajectory using evo:
      - Converts the recorded rosbag to TUM format.
      - Computes Absolute Pose Error (APE) and Relative Pose Error (RPE)
        with alignment.
      - Saves the results and plots.
    """
    
    for name, config in slam_algorithms.items():
        output_bag = os.path.join(output_dir, f"{name}_output.bag")

        print(f"Evaluating trajectory for {algorithm_name}...")
        
        evo_config_cmd = f"evo_config set save_traj_in_zip true"
        subprocess.run(evo_config_cmd, shell=True, check=True)
        # Set evo configuration to save trajectory in zip format
            
        # Compute Absolute Pose Error (APE)
        ape_cmd = (
            f"evo_ape tum {groundtruth_bag} {tum_output} --align "
            f"--save_results {os.path.join(output_dir, algorithm_name + '_ape.zip')} "
            f"--save_plot {os.path.join(output_dir, algorithm_name + '_ape_plot.pdf')}"
        )
        subprocess.run(ape_cmd, shell=True, check=True)
        
        # Compute Relative Pose Error (RPE)
        rpe_cmd = (
            f"evo_rpe tum {groundtruth_bag} {tum_output} --align "
            f"--save_results {os.path.join(output_dir, algorithm_name + '_rpe.zip')} "
            f"--save_plot {os.path.join(output_dir, algorithm_name + '_rpe_plot.pdf')}"
        )
        subprocess.run(rpe_cmd, shell=True, check=True)
        
        print(f"Evaluation for {algorithm_name} completed. Results and plots are saved.")

def process_sequential():
    """
    Process each SLAM algorithm sequentially:
      - Launch the algorithm.
      - Record its output.
      - Play the dataset rosbag.
      - Terminate the processes.
      - Evaluate the recorded trajectory.
    """
    for name, config in slam_algorithms.items():
        print(f"=== Processing {name} sequentially ===")
        
        # Launch SLAM algorithm
        slam_process = launch_slam(name, config)
        time.sleep(5)  # Allow time for initialization
        
        # Start recording the SLAM output
        # output_bag = os.path.join(root_dir, f"{name}_output.bag")
        output_bag = os.path.join(output_dir, f"{name}_output.bag")
        record_process = record_topic(output_bag, config["output_topic"])
        time.sleep(2)  # Delay to ensure recording has started
        
        # Play the dataset rosbag
        play_process = run_command(f"rosbag play {rosbag_full_path} --clock")
        play_process.wait()  # Wait until the rosbag playback is complete
        
        # Terminate recording and SLAM processes
        print("Terminating recording and SLAM process...")
        # record_process.terminate()
        # record_process.send_signal(signal.SIGINT)
        # send SIGINT to the entire process group:
        os.killpg(os.getpgid(record_process.pid), signal.SIGINT)
        record_process.wait()  # Wait for rosbag to finalize the file
        # Optionally, wait a bit more to ensure the file is finalized
        time.sleep(2)
        
        slam_process.terminate()
        slam_process.wait()
        
        print(f"Processing for {name} completed. Output saved to {output_bag}")
        
        # Evaluate the trajectory for the algorithm
        evaluate_trajectory(name, output_bag, config["output_topic"], groundtruth_bag)
        print("----------------------------------------------------------")

def process_concurrent():
    """
    Process all SLAM algorithms concurrently:
      - Launch all SLAM nodes.
      - Start recording for each algorithm.
      - Play the dataset rosbag once.
      - Terminate all processes.
      - Evaluate the recorded trajectories.
    """
    print("=== Processing all SLAM algorithms concurrently ===")
    
    # 1. Enable sim time globally
    # subprocess.run("rosparam set /use_sim_time true", shell=True, check=True)

    # Launch all SLAM algorithms
    slam_processes = {}
    for name, config in slam_algorithms.items():
        print(f"Launching {name}...")
        process = launch_slam(name, config)
        slam_processes[name] = process
        time.sleep(5)  # Allow time for initialization
        
    # Start recording for each SLAM algorithm
    record_processes = {}
    for name, config in slam_algorithms.items():
        output_bag = os.path.join(abs_hierarchy_path, f"{name}_output.bag")
        # output_bag = os.path.join(output_dir, f"{name}_output.bag")
        print(f"Recording {name} output to {output_bag}...")
        process = record_topic(output_bag, config["output_topic"])
        record_processes[name] = process
        time.sleep(2)
    
    # Play the rosbag once for all algorithms
    print(f"Playing dataset bag: {rosbag_full_path}")
    play_process = run_command(f"rosbag play {rosbag_full_path} --clock")
    play_process.wait()  # Wait until playback is finished
    
    # Terminate recording processes
    for name, process in record_processes.items():
        print(f"Stopping recording for {name}...")
        # process.terminate()
        # process.send_signal(signal.SIGINT)
        # process.terminate()
        # send SIGINT to the entire process group:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        # wait until this recorder fully exits
        process.wait()
        # Optionally, wait a bit more to ensure the file is finalized
        time.sleep(2)
    
    # Terminate the SLAM processes
    for name, process in slam_processes.items():
        print(f"Stopping {name}...")
        process.terminate()
        # wait until this SLAM node fully exits
        process.wait()
    
    # ensure absolutely all recorder and SLAM processes have exited
    for p in list(record_processes.values()) + list(slam_processes.values()):
        p.wait()
        
    print("Concurrent processing completed.")

def bag_to_tum():
    """
    Convert the recorded rosbag to TUM format.
    This function is called after all SLAM algorithms have been processed.
    """

    # Now that every process is fully terminated, run evaluations for all SLAM algorithms
    for name, config in slam_algorithms.items():
        output_bag = os.path.join(output_dir, f"{name}_output.bag")
        
        # Remove leading '/' and replace subsequent '/' with '_' in the output topic name
        output_topic = config["output_topic"].lstrip('/').replace("/", "_")
        tum_file =  output_topic + ".tum"        
        # Convert ROS bag trajectory to TUM format
        tum_output = os.path.join(output_dir, f"{name}_trajectory.tum")
        # Note: The bag file must be passed after the flags for evo_traj.
        # traj_cmd = f"evo_traj bag --topic {output_topic} --save_as_tum {tum_output} {output_bag}"
        traj_cmd = f"evo_traj bag {output_bag} {config['output_topic']} --save_as_tum"
        subprocess.run(traj_cmd, shell=True, check=True)
    
        move_file_cmd = f"mv {tum_file} {tum_output}"
        subprocess.run(move_file_cmd, shell=True, check=True)
        print("----------------------------------------------------------")
    
    print(f"Converted {output_bag} to TUM format.")
            
def prepare_and_convert_all(input_dir: str, output_dir: str):
    """
    For each .txt (TUM) file in input_dir:
      1. Make a subfolder in output_dir named <basename>.
      2. Copy the .txt into that folder as 'groundtruth.txt'.
      3. Run `evo_traj tum groundtruth.txt --save_as_bag` there.
      4. Detect the new .bag, rename it to <basename>.bag, keep it in the subfolder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # absolute paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    for tum_path in sorted(glob.glob(os.path.join(input_dir, "*.txt"))):
        base = os.path.splitext(os.path.basename(tum_path))[0]
        subfolder = os.path.join(output_dir, base)
        os.makedirs(subfolder, exist_ok=True)

        # 1) copy & rename txt → groundtruth.txt
        dest_tum = os.path.join(subfolder, "groundtruth.txt")
        print(f"Copying {tum_path} → {dest_tum}")
        shutil.copy2(tum_path, dest_tum)

        # 2) snapshot existing bags
        before = set(glob.glob(os.path.join(subfolder, "*.bag")))

        # 3) convert to bag
        cmd = f"cd {subfolder} && evo_traj tum groundtruth.txt --save_as_bag"
        print(f"Running conversion: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

        # give it a moment
        time.sleep(0.5)

        # 4) detect & rename new bag
        after = set(glob.glob(os.path.join(subfolder, "*.bag")))
        new_bags = after - before
        if not new_bags:
            raise RuntimeError(f"No .bag produced for {base}")
        if len(new_bags) > 1:
            print(f"Warning: multiple new bags in {subfolder}, choosing one arbitrarily.")
        created = new_bags.pop()

        # new_name = base + ".bag"
        new_name =  "groundtruth" + ".bag"
        new_path = os.path.join(subfolder, new_name)
        print(f"Renaming {os.path.basename(created)} → {new_name}")
        os.replace(created, new_path)

    print("All TUM files have been prepared and converted.")

def main():
    parser = argparse.ArgumentParser(description="Automate SLAM processing and evaluation pipeline.")
    parser.add_argument('--mode', choices=['sequential', 'concurrent', 'bag_to_tum', 'tum_to_bag', 'bag_to_hierarchy', 'merge'], required=True,
                        help="Processing mode: 'sequential' for one SLAM at a time, "
                             "'concurrent' for running all SLAM algorithms simultaneously.")
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.mode == 'sequential':
        process_sequential()
        # once all per‐algorithm bags are recorded, merge them and exit
        # merge_bags(abs_hierarchy_path)
        # sys.exit(0)
    elif args.mode == 'concurrent':
        process_concurrent()
        # once all per‐algorithm bags are recorded, merge them and exit
        # merge_bags(abs_hierarchy_path)
        # sys.exit(0)
    elif args.mode == 'bag_to_tum':
        bag_to_tum()
    elif args.mode == 'evaluate':   
        pass
    elif args.mode == 'tum_to_bag':
        prepare_and_convert_all("/media/puonn/zw/puonn/fusion_portable/v2/groundtruth/traj", 
                               "/media/puonn/zw/puonn/fusion_portable/v2/groundtruth/traj/bags")
    elif args.mode == 'merge':
        # Merge all bags in the output directory
        merge_bags(abs_hierarchy_path)
    elif args.mode == 'bag_to_hierarchy':
        examples = [
        "/data/vehicle_multilayer00.bag",
        "/data/ugv_parking03.bag",
        "/data/legged_transition00.bag",
        "/data/handheld_room01.bag",
        ]
        for b in examples:
            print(b, "→", bag_to_hierarchy(b))
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()