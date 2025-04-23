import os
import yaml
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

def read_transform_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    quat = data['quaternion_sensor_body_imu']['data']  # [qw, qx, qy, qz]
    trans = data['translation_sensor_body_imu']['data']  # [x, y, z]
    
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
    rot_matrix = r.as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = trans
    return T

def invert_transform(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv

def save_to_yaml(filepath, T):
    with open(filepath, 'w') as f:
        f.write("transformation_matrix_inverted:\n")
        for row in T:
            row_str = ", ".join(f"{v:.4f}" for v in row)
            f.write(f"\t- [{row_str}]\n")

def main():
    base_path = "/media/puonn/zw/puonn/fusion_portable/v2/calibration_files/"
    yaml_files = glob.glob(os.path.join(base_path, "*_calib_*/calib/frame_cam0[01].yaml")) + \
                    glob.glob(os.path.join(base_path, "*_calib_*/calib/vehicle_frame_cam0[01].yaml"))

    for yaml_file in yaml_files:
        T = read_transform_from_yaml(yaml_file)
        T_inv = invert_transform(T)

        save_path = yaml_file.replace(".yaml", "_inverted.yaml")
        save_to_yaml(save_path, T_inv)
        print(f"Inverted transformation saved to: {save_path}")

if __name__ == "__main__":
    main()
