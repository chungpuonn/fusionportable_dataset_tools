import os
import glob
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_rotmat(quat):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    return r.as_matrix()

def read_transform_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    quat = data['quaternion_sensor_body_imu']['data']
    trans = data['translation_sensor_body_imu']['data']
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat(quat)
    T[:3, 3] = trans
    return T

def invert_transform(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv

def save_inverted_transform_yaml(filepath, T):
    with open(filepath, 'w') as f:
        f.write("transformation_matrix_inverted:\n")
        for row in T:
            row_str = ", ".join(f"{v:.4f}" for v in row)
            f.write(f"\t- [{row_str}]\n")

def process_body_imu_transforms(base_path):
    files = glob.glob(os.path.join(base_path, "*_calib_*/calib/frame_cam0[01].yaml")) + \
                    glob.glob(os.path.join(base_path, "*_calib_*/calib/vehicle_frame_cam0[01].yaml"))
    for yaml_file in files:
        T = read_transform_from_yaml(yaml_file)
        T_inv = invert_transform(T)
        save_path = yaml_file.replace(".yaml", "_inverted.yaml")
        save_inverted_transform_yaml(save_path, T_inv)
        print(f"[✓] Inverted transform saved to: {save_path}")

def read_extrinsics_from_ouster(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    output = {}

    def extract(sensor_key, label):
        q_key = f'quaternion_sensor_{sensor_key}'
        t_key = f'translation_sensor_{sensor_key}'
        if q_key not in data or t_key not in data:
            print(f"[!] Warning: Missing keys for {sensor_key} in {yaml_path}")
            return
        quat = data[q_key]['data']
        trans = data[t_key]['data']
        R_flat = quat_to_rotmat(quat).flatten().tolist()
        output[label] = {
            'camera_ext_R': [round(v, 7) for v in R_flat],
            'camera_ext_t': [round(v, 3) for v in trans]
        }

    # Check and extract all valid ones
    extract("frame_cam00", "frame_cam00")
    extract("frame_cam01", "frame_cam01")
    extract("vehicle_frame_cam00", "vehicle_frame_cam00")
    extract("vehicle_frame_cam01", "vehicle_frame_cam01")

    return output

def save_camera_extrinsics_yaml(data, path):
    with open(path, 'w') as f:
        for cam_key in data.keys():
            R_list = data[cam_key]['camera_ext_R']
            t_list = data[cam_key]['camera_ext_t']
            f.write(f"{cam_key}:\n")
            f.write("  camera_ext_R:\n")
            f.write(f"    [ {R_list[0]: .7f}, {R_list[1]: .7f}, {R_list[2]: .7f},\n")
            f.write(f"     {R_list[3]: .7f}, {R_list[4]: .7f}, {R_list[5]: .7f},\n")
            f.write(f"     {R_list[6]: .7f}, {R_list[7]: .7f}, {R_list[8]: .7f} ]\n")
            f.write(f"  camera_ext_t: [{t_list[0]:.3f}, {t_list[1]:.3f}, {t_list[2]:.3f}]\n")

def process_camera_extrinsics(base_path):
    files = glob.glob(os.path.join(base_path, "*_calib_*/calib/ouster00.yaml"))
    for file in files:
        output_data = read_extrinsics_from_ouster(file)
        out_path = os.path.join(os.path.dirname(file), "camera_lidar_extrinsics_r3live.yaml")
        save_camera_extrinsics_yaml(output_data, out_path)
        print(f"[✓] Camera-Lidar extrinsics saved to: {out_path}")

def main():
    base_path = "/media/puonn/zw/puonn/fusion_portable/v2/calibration_files/"
    print("Processing inverted body-imu transforms...")
    process_body_imu_transforms(base_path)
    print("Processing camera extrinsics from ouster00...")
    process_camera_extrinsics(base_path)

if __name__ == "__main__":
    main()

