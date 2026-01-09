"""
lerobot 0.1.0

converting a dataset(captured from pika sensor) to LeRobot format.

to do:
switch to lerobot v3
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro

import os
from os.path import join as pjoin
import numpy as np
from PIL import Image
import h5py
import cv2

REPO_NAME = "winka9587/pick_cillion_v3"  # Name of the output dataset, also used for the Hugging Face Hub

"""
从指定路径加载图像, 格式如 np.random.randint(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
返回 np.uint8, HxWx3, RGB 格式
"""
def load_image_lerobot(path: str) -> np.ndarray:
    """
    Load an image from path and return it as np.uint8 array (H, W, 3) in RGB format.
    """
    img = Image.open(path).convert("RGB")   # 确保 RGB
    img_np = np.array(img, dtype=np.uint8)  # 转为 numpy
    return img_np

"""
'camera', 
    'color', 
        'pikaDepthCamera_l',    (n, ), [bytes, bytes, ...]
        'pikaDepthCamera_r',    (n, ), [bytes, bytes, ...]
        'pikaFisheyeCamera_l',  (n, ), [bytes, bytes, ...]
        'pikaFisheyeCamera_r'   (n, ), [bytes, bytes, ...]
    'colorExtrinsic', 
        'pikaDepthCamera_l',    (4, 4)
        'pikaDepthCamera_r',    (4, 4)
        'pikaFisheyeCamera_l',  (4, 4)
        'pikaFisheyeCamera_r'   (4, 4)
    'colorIntrinsic', 
        'pikaDepthCamera_l',    (3, 3)
        'pikaDepthCamera_r',    (3, 3)
        'pikaFisheyeCamera_l',  (3, 3)
        'pikaFisheyeCamera_r'   (3, 3)
    'depth', 
        'pikaDepthCamera_l',    (n, ), [bytes, bytes, ...]
        'pikaDepthCamera_r'     (n, ), [bytes, bytes, ...]
    'depthExtrinsic', 
        'pikaDepthCamera_l',    (4, 4)
        'pikaDepthCamera_r'     (4, 4)
    'depthIntrinsic'
        'pikaDepthCamera_l',    (3, 3)
        'pikaDepthCamera_r'     (3, 3)
'gripper', 
    'encoderAngle', 
        'pika_l',               (n, ), [bytes, bytes, ...]
        'pika_r'                (n, ), [bytes, bytes, ...]
    'encoderDistance'
        'pika_l',               (n, ), [bytes, bytes, ...]
        'pika_r'                (n, ), [bytes, bytes, ...]
'instruction', 
'localization', 
    'pose'
        'pika_l'                (n, 6), [bytes, bytes, ...]
        'pika_r'                (n, 6), [bytes, bytes, ...]
'size', 
'timestamp'                     (n, ), [bytes, bytes, ...]
"""
import numpy as np
import cv2

def euler_to_mat(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # R = Rz * Ry * Rx   （根据 roll→pitch→yaw ）
    return Rz @ Ry @ Rx

def poses_rpy_to_vec7_and_actions(poses, *, reference='first', gripper_mode='absolute'):
    """
    poses: ndarray shape (n,6) or (n,7)
           columns: x,y,z, roll, pitch, yaw [, gripper]
    reference: 'first' or 'prev' -- 相对参考（'first' 相对于序列首帧，'prev' 相对于前一帧）
    gripper_mode: 'absolute' (默认) or 'delta'  控制第7列是否为绝对值或 delta
    Returns:
      vec7: (n,7)  -> x,y,z, rotvec(3), gripper
      actions: (n,7) -> translation_delta(3), rotvec_rel(3), gripper (absolute or delta)
    """
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] not in (6, 7):
        raise ValueError("poses must have shape (n,6) or (n,7) with columns x,y,z,roll,pitch,yaw[,gripper]")

    n = poses.shape[0]
    xyz = poses[:, :3]
    rpy = poses[:, 3:6]
    if poses.shape[1] == 7:
        gripper = poses[:, 6].astype(np.float32)
    else:
        gripper = np.zeros((n,), dtype=np.float32)

    # build rotation matrices and rotvecs (axis-angle) for each frame
    R_list = []
    rvec_list = []
    for i in range(n):
        r, p, y = rpy[i]
        R = euler_to_mat(r, p, y)  # 3x3
        R_list.append(R)
        rvec, _ = cv2.Rodrigues(R)  # rvec shape (3,1)
        rvec_list.append(rvec.reshape(3))

    R_arr = np.stack(R_list, axis=0)   # (n,3,3)
    rvec_arr = np.stack(rvec_list, axis=0)  # (n,3)

    # vec7: translation + rotvec + gripper
    vec7 = np.concatenate([xyz, rvec_arr, gripper.reshape(-1,1)], axis=1, dtype=np.float32)

    # compute actions
    actions = np.zeros((n,7), dtype=np.float32)

    if reference == 'first':
        ref_idx = 0
        R_ref = R_arr[ref_idx]      # (3,3)
        trans_ref = xyz[ref_idx]    # (3,)
        grip_ref = gripper[ref_idx]
        for i in range(n):
            # translation delta
            actions[i, :3] = xyz[i] - trans_ref
            # relative rotation: R_rel = R_i * R_ref^T
            R_rel = R_arr[i] @ R_ref.T
            rvec_rel, _ = cv2.Rodrigues(R_rel)
            actions[i, 3:6] = rvec_rel.reshape(3)
            # gripper
            if gripper_mode == 'absolute':
                actions[i, 6] = gripper[i]
            else:  # 'delta'
                actions[i, 6] = gripper[i] - grip_ref

    elif reference == 'prev':
        # first action: zeros or can be seq[0] - ref, here use zeros
        actions[0, :3] = 0.0
        actions[0, 3:6] = 0.0
        actions[0, 6] = gripper[0] if gripper_mode == 'absolute' else 0.0
        for i in range(1, n):
            actions[i, :3] = xyz[i] - xyz[i-1]
            R_rel = R_arr[i] @ R_arr[i-1].T
            rvec_rel, _ = cv2.Rodrigues(R_rel)
            actions[i, 3:6] = rvec_rel.reshape(3)
            if gripper_mode == 'absolute':
                actions[i, 6] = gripper[i]
            else:
                actions[i, 6] = gripper[i] - gripper[i-1]
    else:
        raise ValueError("reference must be 'first' or 'prev'")

    return vec7, actions


def load_pika_hdf5(path: str):
    with h5py.File(path, "r") as f:
        print("all key:", list(f.keys()))  # 'camera', 'gripper', 'instruction', 'localization', 'size', 'timestamp'
        wrist_fisheye_image_l = f['camera']['color']['pikaFisheyeCamera_l'][:]
        wrist_fisheye_image_r = f['camera']['color']['pikaFisheyeCamera_r'][:]
        gripper_angle_l = 1 - (f['gripper']['encoderAngle']['pika_l'][:]/1.67)  # 归一化夹爪, 与openpi相同: 0表示完全打开, 1表示完全关闭  https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md
        gripper_angle_r = 1 - (f['gripper']['encoderAngle']['pika_r'][:]/1.67)
        # 将夹爪开闭状态(0~1)映射到0/1
        # gripper_lb = (gripper_angle_l > 0).astype(int)
        # gripper_rb = (gripper_angle_r > 0).astype(int)
        # gripper_state_l = gripper_lb
        # gripper_state_r = gripper_rb
        gripper_state_l = gripper_angle_l
        gripper_state_r = gripper_angle_r

        pose_l = f['localization']['pose']['pika_l'][:]
        pose_r = f['localization']['pose']['pika_r'][:]
        vec_l, action_l = poses_rpy_to_vec7_and_actions(pose_l, reference='first', gripper_mode='absolute')
        vec_r, action_r = poses_rpy_to_vec7_and_actions(pose_r, reference='first', gripper_mode='absolute')

        vec_l[:, -1] = gripper_state_l
        vec_r[:, -1] = gripper_state_r

        # 计算delta -> 填充action

    assert len(wrist_fisheye_image_l) == len(wrist_fisheye_image_r)
    return {
        "size": len(wrist_fisheye_image_l),
        "wrist_fisheye_image_l": wrist_fisheye_image_l, 
        "wrist_fisheye_image_r": wrist_fisheye_image_r, 
        "vec_l": vec_l,
        "vec_r": vec_r,
        "action_l": action_l,
        "action_r": action_r
        }


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=50,
        features={
            "wrist_image_left": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_right": {
                "dtype": "image",
                "shape": (640, 480, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),  # 6d end pose(x, y, z, roll, pitch, yaw) + gripper
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    # 获取data_dir目录下的所有episode0, episode1, ... 文件夹名
    raw_pika_dataset = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(pjoin(data_dir, f))
    ]
    for episode in raw_pika_dataset:
        episode_data = load_pika_hdf5(pjoin(data_dir, episode, "data.hdf5"))
        len_data = episode_data['size']
        for i in range(len_data):
            prompt = 'put the bottle in the box'
            wrist_image_l = cv2.imread(pjoin(data_dir, episode, episode_data['wrist_fisheye_image_l'][i].decode()))
            wrist_image_l = cv2.rotate(wrist_image_l, cv2.ROTATE_90_CLOCKWISE)
            wrist_image_r = cv2.imread(pjoin(data_dir, episode, episode_data['wrist_fisheye_image_r'][i].decode()))
            wrist_image_r = cv2.rotate(wrist_image_r, cv2.ROTATE_90_CLOCKWISE)
            dataset.add_frame(
                {
                    "wrist_image_left": wrist_image_l,
                    "wrist_image_right": wrist_image_r,
                    "state": episode_data['vec_r'][i],
                    "actions": episode_data["action_r"][i],
                    "prompt": prompt,
                })
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Check output path: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
