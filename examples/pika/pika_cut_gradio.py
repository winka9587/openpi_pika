"""
基于 Gradio 的 Pika 数据裁剪和转换工具

功能：
1. 加载多个 episode 的 data.hdf5 文件
2. 为每个 episode 手动设置起止帧进行裁剪
3. 保存裁剪后的 data_cut.hdf5 文件
4. 保存 lerobot 格式的 dataset
"""

import os
import shutil
from os.path import join as pjoin
from typing import Dict, List, Tuple, Optional
import json
import hashlib
from datetime import datetime

import cv2
import numpy as np
import h5py
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


prompt = "pick the box"  # TODO 修改为其他内容或添加input
name_repo = "winka9587/pick_cillion_v3"  # 默认会在~/.cache/huggingface/lerobot/路径下创建~/.cache/huggingface/lerobot/winka9587/pick_cillion_v3

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


def poses_rpy_to_vec7_and_actions(poses, grippers, *, reference='first', gripper_mode='absolute'):
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
        #TODO
        gripper = grippers
        # gripper = np.zeros((n,), dtype=np.float32)

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
    elif reference == 'current_state':
        # action是当前帧的state 值
        for i in range(n):
            actions[i, :3] = xyz[i]
            actions[i, 3:6] = rvec_arr[i]
            actions[i, 6] = gripper[i]
    else:
        raise ValueError("reference must be 'first' or 'prev'")

    return vec7, actions


def normalize_gripper_angle(gripper_angle, preset_value=1.67, binary=True):
    gripper_angle_open = gripper_angle.max()
    open_threshold = max(gripper_angle_open, preset_value)
    gripper_angle_norm = 1 - (gripper_angle/open_threshold)
    if binary:
        return np.round(gripper_angle_norm)
    else:
        return gripper_angle_norm


def load_pika_hdf5(path: str):
    with h5py.File(path, "r") as f:
        gripper_state_b = True  # gripper只有0和1两种状态
        print("all key:", list(f.keys()))  # 'camera', 'gripper', 'instruction', 'localization', 'size', 'timestamp'
        wrist_fisheye_image_l = f['camera']['color']['pikaFisheyeCamera_l'][:]
        wrist_fisheye_image_r = f['camera']['color']['pikaFisheyeCamera_r'][:]
        

        # gripper_angle_l = 1 - (f['gripper']['encoderAngle']['pika_l'][:]/1.67)  # 归一化夹爪, 与openpi相同: 0表示完全打开, 1表示完全关闭  https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md
        # gripper_angle_r = 1 - (f['gripper']['encoderAngle']['pika_r'][:]/1.67)

        # 保存原始 encoderAngle 值（未归一化的）
        gripper_encoder_angle_l = f['gripper']['encoderAngle']['pika_l'][:]
        gripper_encoder_angle_r = f['gripper']['encoderAngle']['pika_r'][:]
        
        # 归一化后的夹爪值（修改后的值）
        gripper_angle_l = normalize_gripper_angle(gripper_encoder_angle_l)  # 归一化夹爪, 与openpi相同: 0表示完全打开, 1表示完全关闭  https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md
        gripper_angle_r = normalize_gripper_angle(gripper_encoder_angle_r) 

        gripper_state_l = gripper_angle_l
        gripper_state_r = gripper_angle_r

        pose_l = f['localization']['pose']['pika_l'][:]
        pose_r = f['localization']['pose']['pika_r'][:]
        vec_l, action_l = poses_rpy_to_vec7_and_actions(pose_l, gripper_state_l, reference='current_state', gripper_mode='absolute')
        vec_r, action_r = poses_rpy_to_vec7_and_actions(pose_r, gripper_state_r, reference='current_state', gripper_mode='absolute')

        assert vec_l[:, -1].max() <= 1 and vec_l[:, -1].min() == 0
        assert vec_r[:, -1].max() <= 1 and vec_r[:, -1].min() == 0
        assert action_l[:, -1].max() <= 1 and action_l[:, -1].min() == 0
        assert action_r[:, -1].max() <= 1 and action_r[:, -1].min() == 0

        vec_l[:, -1] = gripper_state_l
        vec_r[:, -1] = gripper_state_r

    assert len(wrist_fisheye_image_l) == len(wrist_fisheye_image_r)
    return {
        "size": len(wrist_fisheye_image_l),
        "wrist_fisheye_image_l": wrist_fisheye_image_l, 
        "wrist_fisheye_image_r": wrist_fisheye_image_r, 
        "vec_l": vec_l,
        "vec_r": vec_r,
        "action_l": action_l,
        "action_r": action_r,
        "gripper_encoder_angle_l": gripper_encoder_angle_l,  # 原始 encoderAngle 值（未归一化）
        "gripper_encoder_angle_r": gripper_encoder_angle_r,  # 原始 encoderAngle 值（未归一化）
        }


# 简单缓存，避免重复加载同一个 episode
EPISODE_CACHE = {}

# 预览图像缓存，避免重复加载相同帧
PREVIEW_CACHE = {}


def get_processing_record_path(output_path: str) -> str:
    """获取处理记录JSON文件的路径（保存在output目录下）"""
    return pjoin(output_path, ".pika_processing_records.json")


def load_processing_records(output_path: str, data_dir: str = None) -> Dict:
    """加载处理记录（优先从output_path加载，如果不存在则从data_dir加载以兼容旧数据）"""
    # 优先从output_path加载
    record_path = get_processing_record_path(output_path)
    if os.path.isfile(record_path):
        try:
            with open(record_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"从output_path加载处理记录失败: {e}")
    
    # 如果output_path不存在，尝试从data_dir加载（兼容旧数据）
    if data_dir:
        old_record_path = pjoin(data_dir, ".pika_processing_records.json")
        if os.path.isfile(old_record_path):
            try:
                with open(old_record_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    # 迁移到output_path
                    save_processing_records(output_path, records)
                    print(f"已从data_dir迁移处理记录到output_path")
                    return records
            except Exception as e:
                print(f"从data_dir加载处理记录失败: {e}")
    
    return {}


def save_processing_records(output_path: str, records: Dict):
    """保存处理记录（保存到output目录下）"""
    record_path = get_processing_record_path(output_path)
    try:
        # 确保目录存在
        os.makedirs(output_path, exist_ok=True)
        
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            # 强制刷新文件系统缓存
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"保存处理记录失败: {e}")
        import traceback
        traceback.print_exc()

def generate_unique_episode_name(data_dir: str, original_episode: str, frame_start: int, frame_end: int) -> Tuple[str, str]:
    """生成唯一的episode名称
    
    Args:
        data_dir: 数据目录路径
        original_episode: 原始episode名称
        frame_start: 起始帧
        frame_end: 结束帧
    
    Returns:
        (unique_episode_name, hash_suffix): 唯一episode名称和hash后缀
    """
    # 使用episode目录下的camera/color/pikaFisheyeCamera_l目录中的第一张和最后一张图像来生成哈希
    episode_path = pjoin(data_dir, original_episode)
    camera_dir = pjoin(episode_path, "camera", "color", "pikaFisheyeCamera_l")
    
    hash_input = ""
    try:
        if os.path.isdir(camera_dir):
            # 获取所有图像文件（常见的图像格式）
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [
                f for f in os.listdir(camera_dir)
                if os.path.isfile(pjoin(camera_dir, f)) and 
                os.path.splitext(f.lower())[1] in image_extensions
            ]
            
            if len(image_files) >= 2:
                # 按文件名排序
                image_files.sort()
                first_image = image_files[0]
                last_image = image_files[-1]
                
                # 使用第一张和最后一张图像的文件名、大小和修改时间来生成哈希
                first_path = pjoin(camera_dir, first_image)
                last_path = pjoin(camera_dir, last_image)
                
                first_stat = os.stat(first_path)
                last_stat = os.stat(last_path)
                
                # 组合文件名、大小和修改时间作为哈希输入
                hash_input = f"{first_image}_{first_stat.st_size}_{first_stat.st_mtime}_{last_image}_{last_stat.st_size}_{last_stat.st_mtime}"
            elif len(image_files) == 1:
                # 如果只有一张图像，使用它两次
                image_path = pjoin(camera_dir, image_files[0])
                image_stat = os.stat(image_path)
                hash_input = f"{image_files[0]}_{image_stat.st_size}_{image_stat.st_mtime}_{image_files[0]}_{image_stat.st_size}_{image_stat.st_mtime}"
            else:
                # 如果没有图像文件，回退到使用文件夹创建时间
                creation_time = os.path.getctime(episode_path)
                hash_input = f"no_images_{creation_time}"
                print(f"警告: episode {original_episode} 的 camera/color/pikaFisheyeCamera_l 目录中没有找到图像文件，使用文件夹创建时间")
        else:
            # 如果目录不存在，回退到使用文件夹创建时间
            creation_time = os.path.getctime(episode_path)
            hash_input = f"no_camera_dir_{creation_time}"
            print(f"警告: episode {original_episode} 的 camera/color/pikaFisheyeCamera_l 目录不存在，使用文件夹创建时间")
    except (OSError, ValueError) as e:
        # 如果出现任何错误，回退到使用文件夹创建时间
        try:
            creation_time = os.path.getctime(episode_path)
            hash_input = f"error_{creation_time}"
            print(f"警告: 无法获取图像信息生成哈希，使用文件夹创建时间: {e}")
        except Exception:
            # 最后的回退：使用当前时间
            hash_input = f"fallback_{datetime.now().isoformat()}"
            print(f"警告: 无法获取任何时间信息，使用当前时间")
    
    # 创建哈希以确保唯一性
    hash_str = f"{original_episode}_{hash_input}"
    hash_obj = hashlib.md5(hash_str.encode())
    hash_suffix = hash_obj.hexdigest()[:8]
    return f"{original_episode}_cut_{frame_start}_{frame_end}_{hash_suffix}", hash_suffix


def rename_episode_directory(data_dir: str, old_episode: str, new_episode: str) -> bool:
    """重命名episode目录"""
    try:
        old_path = pjoin(data_dir, old_episode)
        new_path = pjoin(data_dir, new_episode)
        if os.path.isdir(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            return True
        return False
    except Exception as e:
        print(f"重命名episode目录失败: {e}")
        return False


def list_episodes(data_dir: str) -> List[str]:
    """列出 data_dir 下按 episode 数字顺序排序的目录"""
    if not data_dir or not os.path.isdir(data_dir):
        return []

    eps = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(pjoin(data_dir, d)) and d.startswith("episode")
    ]

    # 按 episode 后面的数字排序
    eps.sort(key=lambda x: int(x[len("episode"):]))
    return eps


def get_episode_data(data_dir: str, episode: str) -> Dict:
    """加载并缓存单个 episode 的 data.hdf5"""
    key = (data_dir, episode)
    if key in EPISODE_CACHE:
        return EPISODE_CACHE[key]

    h5_path = pjoin(data_dir, episode, "data.hdf5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"找不到 {h5_path}")

    episode_data = load_pika_hdf5(h5_path)
    EPISODE_CACHE[key] = episode_data
    return episode_data


def is_episode_processed(output_path: str, episode: str, data_dir: str = None) -> bool:
    """检查episode是否已被处理"""
    ep_hash = generate_unique_episode_name(data_dir, episode, 0, 0)[-1]
    records = load_processing_records(output_path, data_dir)
    for record_key, record_info in records.items():
        if record_info.get("original_hash") == ep_hash:
            return True, record_key
    return False, None


def abort_episode(data_dir: str, episode: str, output_path: str) -> str:
    """放弃当前episode，将其移动到output同级目录下的_aborted目录"""
    logs = []
    
    if not data_dir or not os.path.isdir(data_dir):
        return "data_dir 目录无效，请检查"
    
    if not output_path:
        return "output_path 为空，请指定输出目录"
    
    if not episode:
        return "请先选择 episode"
    
    try:
        # 获取output_path的父目录
        output_parent = os.path.dirname(os.path.abspath(output_path))
        if not output_parent:
            # 如果output_path是根目录，使用output_path本身
            output_parent = os.path.dirname(os.path.abspath(output_path)) or output_path
        
        # 创建_aborted目录
        aborted_dir = pjoin(output_parent, "_aborted")
        os.makedirs(aborted_dir, exist_ok=True)
        
        # 移动episode目录到_aborted目录
        original_ep_dir = pjoin(data_dir, episode)
        if os.path.isdir(original_ep_dir):
            target_aborted_dir = pjoin(aborted_dir, episode)
            
            # 如果目标目录已存在，先删除
            if os.path.exists(target_aborted_dir):
                try:
                    shutil.rmtree(target_aborted_dir)
                    logs.append(f"[{episode}] 已清理已存在的目标目录: {target_aborted_dir}")
                except Exception as e:
                    logs.append(f"[{episode}] 清理目标目录失败: {e}")
            
            try:
                # 移动原始目录到_aborted目录
                shutil.move(original_ep_dir, target_aborted_dir)
                logs.append(f"[{episode}] 已移动到放弃目录: {target_aborted_dir}")
            except Exception as e:
                logs.append(f"[{episode}] 移动失败: {e}")
                # 如果移动失败，尝试复制
                try:
                    shutil.copytree(original_ep_dir, target_aborted_dir)
                    logs.append(f"[{episode}] 已复制到放弃目录: {target_aborted_dir}（移动失败，使用复制）")
                except Exception as copy_e:
                    logs.append(f"[{episode}] 复制也失败: {copy_e}")
        else:
            logs.append(f"[{episode}] 警告: 原始数据目录不存在: {original_ep_dir}")
        
    except Exception as e:
        return f"[{episode}] 放弃操作失败: {e}"
    
    return "\n".join(logs)


def refresh_episode_list(data_dir: str, output_path: str = None, filter_processed: bool = True):
    """刷新 episode 列表，可选择过滤已处理的序列"""
    eps = list_episodes(data_dir)
    if len(eps) == 0:
        return gr.update(choices=[], value=None), "未在该目录下找到 episode0, episode1, ... 等子目录"
    
    # 如果启用过滤，移除已处理的序列
    if filter_processed and output_path:
        records = load_processing_records(output_path, data_dir)
        if len(records) > 0:
            # 获取episode的hash
            eps_hash_list = {ep: generate_unique_episode_name(data_dir, ep, 0, 0)[-1] for ep in eps}
            processed_episodes = {record_info.get("original_hash") for record_info in records.values()}
            eps = [ep for ep in eps_hash_list if eps_hash_list[ep] not in processed_episodes]
    
    if len(eps) == 0:
        return gr.update(choices=[], value=None), "所有 episode 都已处理完成，或未找到未处理的 episode"
    
    return gr.update(choices=eps, value=eps[0] if eps else None), f"找到 {len(eps)} 个未处理的 episode: {', '.join(eps)}"


def preview_frame(data_dir: str, episode: str, frame_idx: int, camera_side: str, use_cache: bool = True):
    """预览某一帧图像，支持缓存"""
    try:
        if not data_dir or not os.path.isdir(data_dir):
            return None, gr.update(maximum=0, value=0), "data_dir 目录无效"

        if not episode:
            return None, gr.update(maximum=0, value=0), "请先选择 episode"

        try:
            ep_data = get_episode_data(data_dir, episode)
        except Exception as e:
            return None, gr.update(maximum=0, value=0), f"加载 episode 失败: {e}"

        n = ep_data["size"]
        if n <= 0:
            return None, gr.update(maximum=0, value=0), "该 episode 中没有数据"

        frame_idx = int(frame_idx)
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= n:
            frame_idx = n - 1

        # 检查缓存
        cache_key = (data_dir, episode, frame_idx, camera_side)
        if use_cache and cache_key in PREVIEW_CACHE:
            img_rgb, info = PREVIEW_CACHE[cache_key]
            return img_rgb, gr.update(maximum=n - 1, value=frame_idx), info

        # 获取图像路径
        try:
            if camera_side == "wrist_fisheye_image_l":
                rel_path = ep_data["wrist_fisheye_image_l"][frame_idx].decode()
            else:
                rel_path = ep_data["wrist_fisheye_image_r"][frame_idx].decode()
        except (IndexError, KeyError) as e:
            return None, gr.update(maximum=n - 1, value=frame_idx), f"无法获取图像路径: {e}"

        img_path = pjoin(data_dir, episode, rel_path)
        if not os.path.isfile(img_path):
            return None, gr.update(maximum=n - 1, value=frame_idx), f"找不到图像文件: {img_path}"

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None, gr.update(maximum=n - 1, value=frame_idx), f"无法读取图像: {img_path}"

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 可选：降低预览图像分辨率以提高性能
        # 如果图像太大，可以缩放
        max_preview_size = 640  # 最大预览尺寸
        h, w = img_rgb.shape[:2]
        if max(h, w) > max_preview_size:
            scale = max_preview_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        info = f"episode={episode}, frame={frame_idx}/{n-1}, side={camera_side}, path={rel_path}"
        
        # 缓存结果（限制缓存大小，避免内存溢出）
        if use_cache:
            if len(PREVIEW_CACHE) > 100:  # 限制缓存大小为100张图像
                # 删除最旧的缓存项（简单策略：删除第一个）
                PREVIEW_CACHE.pop(next(iter(PREVIEW_CACHE)))
            PREVIEW_CACHE[cache_key] = (img_rgb, info)
        
        return img_rgb, gr.update(maximum=n - 1, value=frame_idx), info
    
    except Exception as e:
        import traceback
        error_msg = f"预览帧时发生错误: {e}\n{traceback.format_exc()}"
        return None, gr.update(maximum=0, value=0), error_msg


def get_episode_info(data_dir: str, episode: str):
    """获取 episode 的基本信息（总帧数等）"""
    if not data_dir or not episode:
        return "请先选择 episode"
    
    try:
        ep_data = get_episode_data(data_dir, episode)
        n = ep_data["size"]
        return f"Episode: {episode}\n总帧数: {n}\n"
    except Exception as e:
        return f"加载失败: {e}"


def plot_trajectory(
    data_dir: str,
    episode: str,
    frame_start: int = 0,
    frame_end: int = -1,
    arm_side: str = "r",
    pos_outlier_percentile: float = 95.0,
    rot_outlier_percentile: float = 95.0,
    show_position: bool = True,
    show_rotation: bool = True,
    show_gripper: bool = True,
    show_pos_velocity: bool = True,
    show_rot_velocity: bool = True,
    show_xy_projection: bool = True,
    show_xz_projection: bool = True,
    show_yz_projection: bool = True,
) -> Optional[np.ndarray]:
    """
    绘制轨迹可视化图，显示位置、旋转和gripper状态
    
    Args:
        data_dir: 数据目录
        episode: episode 名称
        frame_start: 起始帧
        frame_end: 结束帧（-1 表示最后一帧）
        arm_side: 'l' 或 'r'，选择左臂或右臂
        pos_outlier_percentile: 位置向量离群点阈值（百分位数，0-100）
        rot_outlier_percentile: 旋转向量离群点阈值（百分位数，0-100）
        show_position: 是否显示位置图
        show_rotation: 是否显示旋转向量图
        show_gripper: 是否显示gripper状态图
        show_pos_velocity: 是否显示位置速度图
        show_rot_velocity: 是否显示旋转速度图
        show_xy_projection: 是否显示X-Y投影
        show_xz_projection: 是否显示X-Z投影
        show_yz_projection: 是否显示Y-Z投影
    
    Returns:
        numpy array 格式的图像，如果失败则返回 None
    """
    try:
        if not data_dir or not episode:
            return None
        
        ep_data = get_episode_data(data_dir, episode)
        n = ep_data["size"]
        
        if n <= 0:
            return None
        
        # 规范化帧区间
        s = max(0, int(frame_start))
        e = int(frame_end)
        if e < 0 or e >= n:
            e = n - 1
        if s > e:
            return None
        
        # 获取轨迹数据
        if arm_side == "l":
            vec = ep_data.get("vec_l", None)
            action = ep_data.get("action_l", None)
            # 原始 encoderAngle 值（未归一化）
            gripper_encoder_angle = ep_data.get("gripper_encoder_angle_l", None)
        else:
            vec = ep_data.get("vec_r", None)
            action = ep_data.get("action_r", None)
            # 原始 encoderAngle 值（未归一化）
            gripper_encoder_angle = ep_data.get("gripper_encoder_angle_r", None)
        
        if vec is None:
            return None
        
        # 提取裁剪区间的数据
        vec_cut = vec[s:e+1]
        if action is not None:
            action_cut = action[s:e+1]
        else:
            action_cut = None
        
        # 提取修改后的夹爪值（normalize_gripper_angle 后的值，即 vec 的最后一维）
        gripper_modified = vec_cut[:, -1]
        
        # 提取原始 encoderAngle 数据
        gripper_encoder_angle_cut = None
        if gripper_encoder_angle is not None:
            gripper_encoder_angle_cut = gripper_encoder_angle[s:e+1]
        
        # 计算需要显示的子图数量
        plot_list = []
        if show_position:
            plot_list.append('position')
        if show_rotation:
            plot_list.append('rotation')
        if show_gripper:
            plot_list.append('gripper')
        if show_pos_velocity:
            plot_list.append('pos_velocity')
        if show_rot_velocity:
            plot_list.append('rot_velocity')
        if show_xy_projection:
            plot_list.append('xy_proj')
        if show_xz_projection:
            plot_list.append('xz_proj')
        if show_yz_projection:
            plot_list.append('yz_proj')
        
        num_plots = len(plot_list)
        if num_plots == 0:
            return None
        
        # 计算布局：尽量使用2列
        nrows = (num_plots + 1) // 2
        ncols = 2
        
        # 创建图形
        fig = Figure(figsize=(14, 4 * nrows))
        axes = fig.subplots(nrows, ncols)
        if num_plots == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Trajectory Visualization: {episode} (frames {s}-{e})', fontsize=14, fontweight='bold')
        
        frame_indices = np.arange(s, e + 1)
        
        # 计算离群点（如果任何投影图需要）
        pos_velocity = None
        rot_velocity = None
        pos_threshold = None
        rot_threshold = None
        outlier_indices = None
        
        if show_pos_velocity or show_xy_projection or show_xz_projection or show_yz_projection:
            if len(vec_cut) > 1:
                pos_diff = np.diff(vec_cut[:, :3], axis=0)
                pos_velocity = np.linalg.norm(pos_diff, axis=1)
                pos_threshold = np.percentile(pos_velocity, pos_outlier_percentile)
                outliers = pos_velocity > pos_threshold
                outlier_indices = np.where(outliers)[0] + 1  # +1 因为diff后索引偏移
        
        if show_rot_velocity:
            if len(vec_cut) > 1:
                rot_diff = np.diff(vec_cut[:, 3:6], axis=0)
                rot_velocity = np.linalg.norm(rot_diff, axis=1)
                rot_threshold = np.percentile(rot_velocity, rot_outlier_percentile)
        
        plot_idx = 0
        
        # 绘制各个子图
        for plot_type in plot_list:
            row = plot_idx // 2
            col = plot_idx % 2
            ax = axes[row, col]
            
            if plot_type == 'position':
                # 位置 (x, y, z)
                ax.plot(frame_indices, vec_cut[:, 0], 'r-', label='x', linewidth=1.5, alpha=0.7)
                ax.plot(frame_indices, vec_cut[:, 1], 'g-', label='y', linewidth=1.5, alpha=0.7)
                ax.plot(frame_indices, vec_cut[:, 2], 'b-', label='z', linewidth=1.5, alpha=0.7)
                ax.axvline(x=s, color='orange', linestyle='--', linewidth=2, label='Start', alpha=0.8)
                ax.axvline(x=e, color='red', linestyle='--', linewidth=2, label='End', alpha=0.8)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Position (m)')
                ax.set_title('Position (x, y, z)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'rotation':
                # 旋转向量 (rotvec)
                ax.plot(frame_indices, vec_cut[:, 3], 'r-', label='rx', linewidth=1.5, alpha=0.7)
                ax.plot(frame_indices, vec_cut[:, 4], 'g-', label='ry', linewidth=1.5, alpha=0.7)
                ax.plot(frame_indices, vec_cut[:, 5], 'b-', label='rz', linewidth=1.5, alpha=0.7)
                ax.axvline(x=s, color='orange', linestyle='--', linewidth=2, label='Start', alpha=0.8)
                ax.axvline(x=e, color='red', linestyle='--', linewidth=2, label='End', alpha=0.8)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Rotation Vector')
                ax.set_title('Rotation Vector (rx, ry, rz)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'gripper':
                # Gripper 状态 - 根据当前视角显示对应的原始和修改后的夹爪值
                
                # 绘制原始 encoderAngle 值（未归一化）
                if gripper_encoder_angle_cut is not None:
                    ax.plot(frame_indices, gripper_encoder_angle_cut, 'purple', linewidth=2, alpha=0.8, marker='o', markersize=3, label=f'Original EncoderAngle ({arm_side.upper()})')
                
                # 绘制修改后的夹爪值（normalize_gripper_angle 后的值）
                ax.plot(frame_indices, gripper_modified, 'blue', linewidth=2, alpha=0.7, linestyle='--', marker='s', markersize=2, label=f'Modified (Normalized) ({arm_side.upper()})')
                
                ax.axvline(x=s, color='orange', linestyle='--', linewidth=2, label='Start', alpha=0.8)
                ax.axvline(x=e, color='red', linestyle='--', linewidth=2, label='End', alpha=0.8)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Gripper Value')
                ax.set_title(f'Gripper State Comparison ({arm_side.upper()})')
                # 如果显示原始 encoderAngle，y 轴范围需要调整
                if gripper_encoder_angle_cut is not None:
                    y_min = min(gripper_encoder_angle_cut.min(), gripper_modified.min()) - 0.1
                    y_max = max(gripper_encoder_angle_cut.max(), gripper_modified.max()) + 0.1
                    ax.set_ylim(y_min, y_max)
                else:
                    ax.set_ylim(-0.1, 1.1)
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'pos_velocity':
                # 位置变化率（用于检测抖动）
                if len(vec_cut) > 1 and pos_velocity is not None:
                    ax.plot(frame_indices[1:], pos_velocity, 'c-', linewidth=1.5, alpha=0.7, label='Velocity')
                    # 标记异常高的速度（可能是离群点）
                    if outlier_indices is not None and len(outlier_indices) > 0:
                        ax.scatter(frame_indices[outlier_indices], pos_velocity[outlier_indices - 1], 
                                  color='red', s=50, marker='x', label='Outliers', zorder=5)
                    ax.axhline(y=pos_threshold, color='orange', linestyle=':', linewidth=1.5, 
                              label=f'{pos_outlier_percentile:.1f}th percentile ({pos_threshold:.4f})', alpha=0.7)
                ax.axvline(x=s, color='orange', linestyle='--', linewidth=2, label='Start', alpha=0.8)
                ax.axvline(x=e, color='red', linestyle='--', linewidth=2, label='End', alpha=0.8)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Position Velocity (m/frame)')
                ax.set_title('Position Velocity (detect outliers)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'rot_velocity':
                # 旋转变化率（用于检测抖动）
                if len(vec_cut) > 1 and rot_velocity is not None:
                    ax.plot(frame_indices[1:], rot_velocity, 'm-', linewidth=1.5, alpha=0.7, label='Angular Velocity')
                    # 标记异常高的角速度
                    outliers = rot_velocity > rot_threshold
                    if np.any(outliers):
                        ax.scatter(frame_indices[1:][outliers], rot_velocity[outliers], 
                                  color='red', s=50, marker='x', label='Outliers', zorder=5)
                    ax.axhline(y=rot_threshold, color='orange', linestyle=':', linewidth=1.5, 
                              label=f'{rot_outlier_percentile:.1f}th percentile ({rot_threshold:.4f})', alpha=0.7)
                ax.axvline(x=s, color='orange', linestyle='--', linewidth=2, label='Start', alpha=0.8)
                ax.axvline(x=e, color='red', linestyle='--', linewidth=2, label='End', alpha=0.8)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel('Angular Velocity')
                ax.set_title('Rotation Velocity (detect outliers)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            elif plot_type == 'xy_proj':
                # 3D 轨迹投影 (X-Y)
                ax.plot(vec_cut[:, 0], vec_cut[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
                ax.scatter(vec_cut[0, 0], vec_cut[0, 1], color='green', s=100, marker='o', 
                          label='Start', zorder=5, edgecolors='black', linewidths=2)
                ax.scatter(vec_cut[-1, 0], vec_cut[-1, 1], color='red', s=100, marker='s', 
                          label='End', zorder=5, edgecolors='black', linewidths=2)
                # 标记异常点
                if outlier_indices is not None and len(outlier_indices) > 0:
                    ax.scatter(vec_cut[outlier_indices, 0], vec_cut[outlier_indices, 1],
                              color='red', s=80, marker='x', label='Outliers', zorder=5, linewidths=2)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('2D Trajectory Projection (X-Y)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            
            elif plot_type == 'xz_proj':
                # 3D 轨迹投影 (X-Z)
                ax.plot(vec_cut[:, 0], vec_cut[:, 2], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
                ax.scatter(vec_cut[0, 0], vec_cut[0, 2], color='green', s=100, marker='o', 
                          label='Start', zorder=5, edgecolors='black', linewidths=2)
                ax.scatter(vec_cut[-1, 0], vec_cut[-1, 2], color='red', s=100, marker='s', 
                          label='End', zorder=5, edgecolors='black', linewidths=2)
                # 标记异常点
                if outlier_indices is not None and len(outlier_indices) > 0:
                    ax.scatter(vec_cut[outlier_indices, 0], vec_cut[outlier_indices, 2],
                              color='red', s=80, marker='x', label='Outliers', zorder=5, linewidths=2)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title('2D Trajectory Projection (X-Z)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            
            elif plot_type == 'yz_proj':
                # 3D 轨迹投影 (Y-Z)
                ax.plot(vec_cut[:, 1], vec_cut[:, 2], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
                ax.scatter(vec_cut[0, 1], vec_cut[0, 2], color='green', s=100, marker='o', 
                          label='Start', zorder=5, edgecolors='black', linewidths=2)
                ax.scatter(vec_cut[-1, 1], vec_cut[-1, 2], color='red', s=100, marker='s', 
                          label='End', zorder=5, edgecolors='black', linewidths=2)
                # 标记异常点
                if outlier_indices is not None and len(outlier_indices) > 0:
                    ax.scatter(vec_cut[outlier_indices, 1], vec_cut[outlier_indices, 2],
                              color='red', s=80, marker='x', label='Outliers', zorder=5, linewidths=2)
                ax.set_xlabel('Y (m)')
                ax.set_ylabel('Z (m)')
                ax.set_title('2D Trajectory Projection (Y-Z)')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            
            plot_idx += 1
        
        # 隐藏未使用的子图
        for i in range(plot_idx, nrows * ncols):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        fig.tight_layout()
        
        # 转换为 numpy array - 使用兼容的方法
        fig.canvas.draw()
        
        # 方法1: 尝试使用 buffer_rgba (新版本 matplotlib)
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            width, height = fig.canvas.get_width_height()
            buf = buf.reshape((height, width, 4))
            # 转换为 RGB (去掉 alpha 通道)
            buf = buf[:, :, :3]
        except AttributeError:
            # 方法2: 使用 tostring_rgb (旧版本 matplotlib)
            try:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 方法3: 使用 io.BytesIO 和 PIL (最兼容的方法)
                buf_io = io.BytesIO()
                fig.savefig(buf_io, format='png', bbox_inches='tight', dpi=100)
                buf_io.seek(0)
                img = Image.open(buf_io)
                buf = np.array(img)
                # 如果是 RGBA，转换为 RGB
                if buf.shape[2] == 4:
                    buf = buf[:, :, :3]
                buf_io.close()
        
        plt.close(fig)
        
        return buf
    
    except Exception as e:
        import traceback
        print(f"绘制轨迹图失败: {e}\n{traceback.format_exc()}")
        return None


def save_cut_hdf5(
    data_dir: str,
    episode: str,
    output_dir: str,
    frame_start: int,
    frame_end: int,
    unique_episode_name: Optional[str] = None,
) -> str:
    """
    保存裁剪后的 data_cut.hdf5 文件
    
    需要从原始 hdf5 中读取所有数据，然后只保存裁剪区间的数据
    """
    try:
        # 加载原始数据
        ep_data = get_episode_data(data_dir, episode)
        n = ep_data["size"]
        
        # 规范化帧区间
        s = max(0, int(frame_start))
        e = int(frame_end)
        if e < 0 or e >= n:
            e = n - 1
        if s > e:
            return f"[{episode}] 帧区间非法: start={s}, end={e}"
        
        # 读取原始 hdf5 文件
        original_h5_path = pjoin(data_dir, episode, "data.hdf5")
        if not os.path.isfile(original_h5_path):
            return f"[{episode}] 找不到原始文件: {original_h5_path}"
        
        # 创建输出目录（使用唯一名称）
        output_episode_name = unique_episode_name if unique_episode_name else episode
        out_ep_dir = pjoin(output_dir, output_episode_name)
        os.makedirs(out_ep_dir, exist_ok=True)
        output_h5_path = pjoin(out_ep_dir, "data_cut.hdf5")
        
        # 读取原始 hdf5 并写入裁剪后的数据
        with h5py.File(original_h5_path, "r") as f_in:
            with h5py.File(output_h5_path, "w") as f_out:
                # 复制 size（原始格式可能是标量或数组）
                new_size = e - s + 1
                if "size" in f_in:
                    size_dtype = f_in["size"].dtype
                    size_shape = f_in["size"].shape
                    if size_shape == ():
                        f_out.create_dataset("size", data=np.array(new_size, dtype=size_dtype))
                    else:
                        f_out.create_dataset("size", data=np.array([new_size], dtype=size_dtype))
                else:
                    f_out.create_dataset("size", data=np.array(new_size, dtype=np.int64))
                
                # 复制 timestamp（裁剪区间）
                if "timestamp" in f_in:
                    timestamp_ds = f_in["timestamp"]
                    if timestamp_ds.shape == ():  # 标量
                        f_out.create_dataset("timestamp", data=timestamp_ds[()])
                    else:
                        timestamp_data = timestamp_ds[s:e+1]
                        f_out.create_dataset("timestamp", data=timestamp_data)
                
                # 复制 instruction
                if "instruction" in f_in:
                    instruction_ds = f_in["instruction"]
                    if instruction_ds.shape == ():  # 标量
                        f_out.create_dataset("instruction", data=instruction_ds[()])
                    else:
                        instruction_data = instruction_ds[:]
                        f_out.create_dataset("instruction", data=instruction_data)
                
                # 处理 camera 数据
                if "camera" in f_in:
                    cam_grp = f_out.create_group("camera")
                    
                    # color 数据
                    if "color" in f_in["camera"]:
                        color_grp = cam_grp.create_group("color")
                        for cam_name in ["pikaFisheyeCamera_l", "pikaFisheyeCamera_r", 
                                        "pikaDepthCamera_l", "pikaDepthCamera_r"]:
                            if cam_name in f_in["camera"]["color"]:
                                color_ds = f_in["camera"]["color"][cam_name]
                                if color_ds.shape == ():  # 标量
                                    color_grp.create_dataset(cam_name, data=color_ds[()])
                                else:
                                    color_data = color_ds[s:e+1]
                                    color_grp.create_dataset(cam_name, data=color_data)
                    
                    # colorExtrinsic
                    if "colorExtrinsic" in f_in["camera"]:
                        ext_grp = cam_grp.create_group("colorExtrinsic")
                        for cam_name in f_in["camera"]["colorExtrinsic"].keys():
                            ext_data = f_in["camera"]["colorExtrinsic"][cam_name][:]
                            ext_grp.create_dataset(cam_name, data=ext_data)
                    
                    # colorIntrinsic
                    if "colorIntrinsic" in f_in["camera"]:
                        int_grp = cam_grp.create_group("colorIntrinsic")
                        for cam_name in f_in["camera"]["colorIntrinsic"].keys():
                            int_data = f_in["camera"]["colorIntrinsic"][cam_name][:]
                            int_grp.create_dataset(cam_name, data=int_data)
                    
                    # depth 数据
                    if "depth" in f_in["camera"]:
                        depth_grp = cam_grp.create_group("depth")
                        for cam_name in ["pikaDepthCamera_l", "pikaDepthCamera_r"]:
                            if cam_name in f_in["camera"]["depth"]:
                                depth_ds = f_in["camera"]["depth"][cam_name]
                                if depth_ds.shape == ():  # 标量
                                    depth_grp.create_dataset(cam_name, data=depth_ds[()])
                                else:
                                    depth_data = depth_ds[s:e+1]
                                    depth_grp.create_dataset(cam_name, data=depth_data)
                    
                    # depthExtrinsic
                    if "depthExtrinsic" in f_in["camera"]:
                        depth_ext_grp = cam_grp.create_group("depthExtrinsic")
                        for cam_name in f_in["camera"]["depthExtrinsic"].keys():
                            depth_ext_data = f_in["camera"]["depthExtrinsic"][cam_name][:]
                            depth_ext_grp.create_dataset(cam_name, data=depth_ext_data)
                    
                    # depthIntrinsic
                    if "depthIntrinsic" in f_in["camera"]:
                        depth_int_grp = cam_grp.create_group("depthIntrinsic")
                        for cam_name in f_in["camera"]["depthIntrinsic"].keys():
                            depth_int_data = f_in["camera"]["depthIntrinsic"][cam_name][:]
                            depth_int_grp.create_dataset(cam_name, data=depth_int_data)
                
                # 处理 gripper 数据
                if "gripper" in f_in:
                    grip_grp = f_out.create_group("gripper")
                    
                    if "encoderAngle" in f_in["gripper"]:
                        angle_grp = grip_grp.create_group("encoderAngle")
                        for arm_name in ["pika_l", "pika_r"]:
                            if arm_name in f_in["gripper"]["encoderAngle"]:
                                angle_ds = f_in["gripper"]["encoderAngle"][arm_name]
                                if angle_ds.shape == ():  # 标量
                                    angle_grp.create_dataset(arm_name, data=angle_ds[()])
                                else:
                                    angle_data = angle_ds[s:e+1]
                                    angle_grp.create_dataset(arm_name, data=angle_data)
                    
                    if "encoderDistance" in f_in["gripper"]:
                        dist_grp = grip_grp.create_group("encoderDistance")
                        for arm_name in ["pika_l", "pika_r"]:
                            if arm_name in f_in["gripper"]["encoderDistance"]:
                                dist_ds = f_in["gripper"]["encoderDistance"][arm_name]
                                if dist_ds.shape == ():  # 标量
                                    dist_grp.create_dataset(arm_name, data=dist_ds[()])
                                else:
                                    dist_data = dist_ds[s:e+1]
                                    dist_grp.create_dataset(arm_name, data=dist_data)
                
                # 处理 localization 数据
                if "localization" in f_in:
                    loc_grp = f_out.create_group("localization")
                    if "pose" in f_in["localization"]:
                        pose_grp = loc_grp.create_group("pose")
                        for arm_name in ["pika_l", "pika_r"]:
                            if arm_name in f_in["localization"]["pose"]:
                                pose_ds = f_in["localization"]["pose"][arm_name]
                                if pose_ds.shape == ():  # 标量
                                    pose_grp.create_dataset(arm_name, data=pose_ds[()])
                                else:
                                    pose_data = pose_ds[s:e+1]
                                    pose_grp.create_dataset(arm_name, data=pose_data)
        
        return f"[{episode}] 成功保存 data_cut.hdf5 到 {output_h5_path} (帧区间: {s}~{e}, 共 {e-s+1} 帧, 输出名称: {output_episode_name})"
    
    except Exception as e:
        return f"[{episode}] 保存 data_cut.hdf5 失败: {e}"


def gen_cut_hdf5(
    data_dir: str,
    episode: str,
    output_path: str,
    frame_start: int,
    frame_end: int,
    update_records: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    处理单个 episode，保存 data_cut.hdf5 并更新处理记录
    
    Args:
        data_dir: 输入数据目录
        episode: 当前要处理的 episode 名称
        output_path: 输出目录
        frame_start: 起始帧
        frame_end: 结束帧
        update_records: 是否更新处理记录
    
    Returns:
        (log_message, unique_episode_name): 日志消息和唯一episode名称
    """
    logs = []
    unique_episode_name = None
    
    if not data_dir or not os.path.isdir(data_dir):
        return "data_dir 目录无效，请检查", None
    
    if not output_path:
        return "output_path 为空，请指定输出目录", None
    
    if not episode:
        return "请先选择 episode", None
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # 生成唯一episode名称（使用episode文件夹的创建时间）
        unique_episode_name, hash_suffix = generate_unique_episode_name(data_dir, episode, frame_start, frame_end)
        
        # 创建输出目录
        out_ep_dir = pjoin(output_path, unique_episode_name)
        os.makedirs(out_ep_dir, exist_ok=True)
        
        # 保存 data_cut.hdf5
        log_msg = save_cut_hdf5(data_dir, episode, output_path, frame_start, frame_end, unique_episode_name)
        logs.append(log_msg)
        
        # 更新处理记录
        if update_records:
            records = load_processing_records(output_path, data_dir)
            record_key = unique_episode_name
            records[record_key] = {
                "original_episode": episode,
                "original_hash": hash_suffix,
                "unique_name": unique_episode_name,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_count": frame_end - frame_start + 1,
                "processed_at": datetime.now().isoformat(),
                "output_path": output_path,
                "data_cut_path": pjoin(output_path, unique_episode_name, "data_cut.hdf5"),
                "original_data_path": pjoin(output_path, unique_episode_name, episode),
                "in_lerobot": False,  # 标记是否已添加到 lerobot 数据集
            }
            save_processing_records(output_path, records)
            logs.append(f"[{episode}] 已更新处理记录: {record_key}")
    except Exception as e:
        return f"[{episode}] 处理失败: {e}", None
    
    return "\n".join(logs), unique_episode_name


def move_single_episode(
    data_dir: str,
    episode: str,
    output_path: str,
    frame_start: int,
    frame_end: int,
    unique_episode_name: str,
    update_records: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    处理单个 episode，保存 data_cut.hdf5 并更新处理记录
    
    Args:
        data_dir: 输入数据目录
        episode: 当前要处理的 episode 名称
        output_path: 输出目录
        frame_start: 起始帧
        frame_end: 结束帧
        update_records: 是否更新处理记录
    
    Returns:
        (log_message, unique_episode_name): 日志消息和唯一episode名称
    """
    logs = []
    
    if not data_dir or not os.path.isdir(data_dir):
        return "data_dir 目录无效，请检查", None
    
    if not output_path:
        return "output_path 为空，请指定输出目录", None
    
    if not episode:
        return "请先选择 episode", None
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # 创建输出目录
        out_ep_dir = pjoin(output_path, unique_episode_name)
        os.makedirs(out_ep_dir, exist_ok=True)
        
        # 将原始episode目录移动到output目录下
        original_ep_dir = pjoin(data_dir, episode)
        if os.path.isdir(original_ep_dir):
            # 目标路径：将原始目录移动到输出目录下，保持原始名称
            target_original_dir = pjoin(out_ep_dir, episode)
            
            # 如果目标目录已存在，先删除（可能是之前处理失败留下的）
            if os.path.exists(target_original_dir):
                try:
                    shutil.rmtree(target_original_dir)
                    logs.append(f"[{episode}] 已清理已存在的目标目录: {target_original_dir}")
                except Exception as e:
                    logs.append(f"[{episode}] 清理目标目录失败: {e}")
            
            try:
                # 移动原始目录到输出目录下
                shutil.move(original_ep_dir, target_original_dir)
                logs.append(f"[{episode}] 已移动原始数据目录到: {target_original_dir}")
            except Exception as e:
                logs.append(f"[{episode}] 移动原始数据目录失败: {e}")
                # 如果移动失败，尝试复制（作为备选方案）
                try:
                    shutil.copytree(original_ep_dir, target_original_dir)
                    logs.append(f"[{episode}] 已复制原始数据目录到: {target_original_dir}（移动失败，使用复制）")
                except Exception as copy_e:
                    logs.append(f"[{episode}] 复制原始数据目录也失败: {copy_e}")
        else:
            logs.append(f"[{episode}] 警告: 原始数据目录不存在: {original_ep_dir}")
        
        # 更新处理记录
        if update_records:
            records = load_processing_records(output_path, data_dir)
            # records[record_key] = {
            #     "original_episode": episode,
            #     "unique_name": unique_episode_name,
            #     "frame_start": frame_start,
            #     "frame_end": frame_end,
            #     "frame_count": frame_end - frame_start + 1,
            #     "processed_at": datetime.now().isoformat(),
            #     "output_path": output_path,
            #     "data_cut_path": pjoin(output_path, unique_episode_name, "data_cut.hdf5"),
            #     "original_data_path": pjoin(output_path, unique_episode_name, episode),
            #     "in_lerobot": True,  # 标记是否已添加到 lerobot 数据集
            # }
            # save_processing_records(output_path, records)
            record_key = unique_episode_name
            records[record_key]["in_lerobot"] = True
            records[record_key]["lerobot_added_at"] = datetime.now().isoformat()
            logs.append(f"[{episode}] 已更新处理记录: {record_key}")
    except Exception as e:
        return f"[{episode}] 处理失败: {e}", None
    
    return "\n".join(logs), unique_episode_name


def create_lerobot_dataset_from_records(
    data_dir: str,
    output_path: str,
    repo_name: str,
    progress=None,
    clear_dataset=False
) -> str:
    """
    从处理记录中读取所有已处理的序列，创建lerobot格式数据集
    
    Args:
        data_dir: 输入数据目录（包含处理记录JSON）
        output_path: 输出目录
        repo_name: lerobot dataset 的 repo_id
    
    Returns:
        日志消息
    """
    logs = []
    
    if not data_dir or not os.path.isdir(data_dir):
        return "data_dir 目录无效，请检查"
    
    if not output_path:
        return "output_path 为空，请指定输出目录"
    
    # 加载处理记录
    records = load_processing_records(output_path, data_dir)
    if not records:
        return "没有找到处理记录，请先处理一些episode"
    
    logs.append(f"找到 {len(records)} 条处理记录")
    
    lerobot_output_path = HF_LEROBOT_HOME / repo_name
    if clear_dataset and lerobot_output_path.exists():
        shutil.rmtree(lerobot_output_path)

    # 如果 dataset 已存在，加载处理记录以确定已处理的 episode
    processed_episodes_from_records = set()
    lerobot_is_exists = False
    if lerobot_output_path.exists():
        lerobot_is_exists = True
        print(f"dataset exists({lerobot_output_path}), append new episode")
        # 加载处理记录，获取已处理的 episode
        records = load_processing_records(output_path, data_dir)
        processed_episodes_from_records = {record_info.get("original_episode") 
                                          for record_info in records.values()}
        logs.append(f"已加载处理记录，发现 {len(processed_episodes_from_records)} 个已处理的 episode: {sorted(processed_episodes_from_records)}")
        
        dataset = LeRobotDataset(repo_id=repo_name)
        dataset.start_image_writer(
            num_processes=5,
            num_threads=10,
        )
    else:
        # 创建 LeRobot dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="gbt",
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
                    "shape": (7,),
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
    
    try:
        # 处理每个记录
        processed_count = 0
        skipped_count = 0
        total_records = len(records)
        
        for idx, (record_key, record_info) in enumerate(records.items()):
            # 更新进度条
            if progress is not None:
                progress((idx, total_records), desc=f"处理记录 {idx+1}/{total_records}: {record_key}")
            try:
                original_episode = record_info.get("original_episode", "")
                unique_name = record_info.get("unique_name", record_key)
                
                # 如果 dataset 已存在且该记录已标记为在 lerobot 中，跳过
                if lerobot_is_exists and record_info.get("in_lerobot", False):
                    logs.append(f"[{unique_name}] 跳过：已在 lerobot 数据集中 (in_lerobot=True)")
                    skipped_count += 1
                    continue
                
                data_cut_path = record_info.get("data_cut_path", "")
                
                # 如果data_cut_path不存在，尝试从output_path构建
                if not os.path.isfile(data_cut_path):
                    data_cut_path = pjoin(output_path, unique_name, "data_cut.hdf5")
                
                if not os.path.isfile(data_cut_path):
                    logs.append(f"[{unique_name}] 跳过：找不到 data_cut.hdf5: {data_cut_path}")
                    skipped_count += 1
                    continue
                
                # 加载裁剪后的数据
                episode_data = load_pika_hdf5(data_cut_path)
                len_data = episode_data['size']
                
                # 添加每一帧
                for i in range(len_data):
                    # 更新进度条（帧级别）
                    if progress is not None:
                        frame_progress = (i + 1) / len_data
                        progress((idx + frame_progress, total_records), desc=f"处理 {unique_name}: 帧 {i+1}/{len_data}")
                    
                    # 图像路径：优先从output目录下的移动后的目录读取，如果不存在则从原始位置读取
                    original_data_path = record_info.get("original_data_path", "")
                    if original_data_path and os.path.isdir(original_data_path):
                        # 使用移动后的目录
                        img_path_l = pjoin(original_data_path, episode_data['wrist_fisheye_image_l'][i].decode())
                        img_path_r = pjoin(original_data_path, episode_data['wrist_fisheye_image_r'][i].decode())
                    else:
                        # 回退到原始位置
                        img_path_l = pjoin(data_dir, original_episode, episode_data['wrist_fisheye_image_l'][i].decode())
                        img_path_r = pjoin(data_dir, original_episode, episode_data['wrist_fisheye_image_r'][i].decode())
                    
                    wrist_image_l = cv2.imread(img_path_l)
                    if wrist_image_l is None:
                        logs.append(f"[{unique_name}] 帧 {i} 无法读取左图像: {img_path_l}")
                        continue
                    wrist_image_l = cv2.rotate(wrist_image_l, cv2.ROTATE_90_CLOCKWISE)
                    
                    wrist_image_r = cv2.imread(img_path_r)
                    if wrist_image_r is None:
                        logs.append(f"[{unique_name}] 帧 {i} 无法读取右图像: {img_path_r}")
                        continue
                    wrist_image_r = cv2.rotate(wrist_image_r, cv2.ROTATE_90_CLOCKWISE)
                    
                    # TODO 这里应该加载config检查
                    dataset.add_frame({
                        "wrist_image_left": wrist_image_l,
                        "wrist_image_right": wrist_image_r,
                        "state": episode_data['vec_r'][i],
                        "actions": episode_data["action_r"][i],
                        "prompt": prompt,
                    })
                    
                    # 释放图像内存，减少内存峰值
                    del wrist_image_l, wrist_image_r
                
                dataset.save_episode()
                
                # 清理 episode_data 以释放内存
                del episode_data
                
                # 更新处理记录中的 in_lerobot 标志
                if record_key in records:
                    records[record_key]["in_lerobot"] = True
                    records[record_key]["lerobot_added_at"] = datetime.now().isoformat()
                    save_processing_records(output_path, records)
                
                # 处理当前 episode（只保存 data_cut.hdf5 和更新记录，不创建 dataset）
                log_msg, unique_name = move_single_episode(
                    data_dir,
                    original_episode,
                    output_path,
                    record_info.get("frame_start", ""),
                    record_info.get("frame_end", ""),
                    record_info.get("unique_name", ""),
                    update_records=True
                )
                
                # 追加日志而不是替换
                logs.append(f"移动序列:\n{log_msg}")
                
                processed_count += 1
                logs.append(f"[{unique_name}] 已添加到 lerobot dataset ({len_data} 帧, 原始: {original_episode})")
            
            except Exception as e:
                logs.append(f"[{record_key}] 处理失败: {e}")
                import traceback
                logs.append(traceback.format_exc())
                skipped_count += 1
        
        logs.append(f"\n完成！成功处理 {processed_count} 个序列，跳过 {skipped_count} 个序列")
        
        # 移动 dataset 到目标位置
        try:
            if lerobot_output_path.exists() and lerobot_output_path != lerobot_output_path:
                # 如果目标路径已存在，先删除
                if lerobot_output_path.exists():
                    shutil.rmtree(lerobot_output_path)
                # 移动整个目录
                shutil.move(str(lerobot_output_path), str(lerobot_output_path))
                logs.append(f"LeRobot dataset 已移动到: {lerobot_output_path}")
            else:
                logs.append(f"LeRobot dataset 已保存到: {lerobot_output_path}")
        except Exception as move_e:
            logs.append(f"移动 dataset 到目标位置失败: {move_e}")
            logs.append(f"目标 dataset 位置: {lerobot_output_path}")
    
    except Exception as e:
        logs.append(f"\n创建 LeRobot dataset 失败: {e}")
        import traceback
        logs.append(traceback.format_exc())
    
    return "\n".join(logs)


def build_ui():
    """构建 Gradio UI"""
    with gr.Blocks(title="Pika 数据裁剪和转换工具") as demo:
        gr.Markdown(
            """
            # Pika 数据裁剪和转换工具
            
            - 加载多个 episode 的 data.hdf5 文件
            - 为每个 episode 手动设置起止帧进行裁剪
            - 保存裁剪后的 data_cut.hdf5 文件
            - 保存 lerobot 格式的 dataset
            """
        )
        
        gr.Markdown("## 配置选项")
        with gr.Row():
            data_dir = gr.Textbox(
                label="数据根目录 data_dir",
                placeholder="/data/pika_cillion",
                value="/data/pika_cillion"
            )
            output_path = gr.Textbox(
                label="输出目录 output_path",
                placeholder="/data/pika_cillion_v3_output",
                value="/data/pika_cillion_v3_output"
            )
        
        with gr.Row():
            repo_name = gr.Textbox(
                label="LeRobot Repo Name（用于批量创建数据集）",
                value=name_repo,
                placeholder=name_repo,
                info="仅在点击'批量创建 LeRobot 数据集'按钮时使用"
            )
        
        gr.Markdown("### 轨迹可视化图保存设置")
        with gr.Row():
            save_trajectory_image = gr.Checkbox(
                label="保存轨迹可视化图",
                value=True,
                info="处理episode时是否保存轨迹可视化图"
            )
            trajectory_image_path = gr.Textbox(
                label="轨迹图保存路径",
                placeholder="/data/pika_cillion_v3_output/trajectory_images",
                value="",
                info="留空则保存到output_path目录。支持相对路径和绝对路径"
            )
        
        with gr.Row():
            refresh_btn = gr.Button("刷新 episode 列表", variant="primary")
            episode_dd = gr.Dropdown(
                label="Episode 列表（用于预览）",
                choices=[],
                value=None,
                interactive=True
            )
            episode_info = gr.Textbox(
                label="Episode 信息",
                interactive=False,
                lines=3
            )
        
        gr.Markdown("### 处理记录")
        with gr.Row():
            refresh_records_btn = gr.Button("刷新处理记录", variant="secondary")
            processing_records = gr.Textbox(
                label="已处理的序列记录",
                interactive=False,
                lines=10,
                value="点击'刷新处理记录'查看已处理的序列"
            )
        
        gr.Markdown("### 时间轴预览和裁剪设置（拖动滑块设置起止帧）")
        
        with gr.Row():
            camera_side = gr.Radio(
                ["wrist_fisheye_image_l", "wrist_fisheye_image_r"],
                value="wrist_fisheye_image_r",
                label="相机侧",
            )
            arm_side = gr.Radio(
                ["l", "r"],
                value="r",
                label="机械臂侧（轨迹可视化）",
            )
        
        # 使用两个滑块实现范围选择（Gradio 的 Slider 不支持列表作为 value）
        # 使用 live=False 避免拖动时频繁更新，只在释放滑块时更新
        with gr.Row():
            frame_start = gr.Slider(
                label="起始帧",
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                interactive=True,
            )
            frame_end = gr.Slider(
                label="结束帧",
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                interactive=True,
            )
        
        # 添加一个手动更新预览的按钮（可选）
        update_preview_btn = gr.Button("更新预览", variant="secondary")
        
        # 显示当前预览的帧索引（起始帧或结束帧）
        current_preview_frame = gr.State(value=0)  # 存储当前预览的帧索引
        
        # 使用两个窗口分别显示起止帧
        with gr.Row():
            preview_img_start = gr.Image(label="起始帧预览", type="numpy")
            preview_img_end = gr.Image(label="结束帧预览", type="numpy")
        
        preview_info = gr.Textbox(label="预览信息", interactive=False)
        
        # 轨迹可视化图
        gr.Markdown("### 轨迹可视化（用于检查离群点和抖动）")
        with gr.Row():
            pos_outlier_threshold = gr.Number(
                label="位置向量离群点阈值（百分位数）",
                value=95.0,
                minimum=0.0,
                maximum=100.0,
                step=1.0,
                precision=1,
                info="0-100，值越大，标记的离群点越少"
            )
            rot_outlier_threshold = gr.Number(
                label="旋转向量离群点阈值（百分位数）",
                value=95.0,
                minimum=0.0,
                maximum=100.0,
                step=1.0,
                precision=1,
                info="0-100，值越大，标记的离群点越少"
            )
        
        gr.Markdown("#### 可视化选项（默认全开）")
        with gr.Row():
            show_position = gr.Checkbox(label="显示位置图", value=True)
            show_rotation = gr.Checkbox(label="显示旋转向量图", value=True)
            show_gripper = gr.Checkbox(label="显示Gripper状态图", value=True)
            show_pos_velocity = gr.Checkbox(label="显示位置速度图", value=True)
        with gr.Row():
            show_rot_velocity = gr.Checkbox(label="显示旋转速度图", value=True)
            show_xy_projection = gr.Checkbox(label="显示X-Y投影", value=True)
            show_xz_projection = gr.Checkbox(label="显示X-Z投影", value=True)
            show_yz_projection = gr.Checkbox(label="显示Y-Z投影", value=True)
        
        trajectory_plot = gr.Image(label="轨迹可视化图", type="numpy")
        
        # 用于标记是否正在程序切换episode（避免触发on_episode_change加载预览）
        is_programmatic_switch = gr.State(value=False)
        
        gr.Markdown("### 处理操作")
        with gr.Row():
            process_btn = gr.Button("处理当前episode", variant="primary")
            abort_btn = gr.Button("放弃当前episode", variant="stop")
            batch_create_btn = gr.Button("批量创建 LeRobot 数据集（从处理记录）", variant="secondary")
        
        # 进度条（用于批量创建数据集）
        batch_progress = gr.Progress()
        
        process_log = gr.Textbox(label="处理日志", lines=20)
        
        def refresh_processing_records(data_dir_val, output_path_val):
            # 清空缓存
            EPISODE_CACHE.clear()
            PREVIEW_CACHE.clear()   
            """刷新处理记录显示"""
            if not output_path_val:
                return "请先设置output_path"
            
            records = load_processing_records(output_path_val, data_dir_val)
            if not records:
                return "暂无处理记录"
            
            lines = [f"共 {len(records)} 条处理记录：\n"]
            lines.append("=" * 80)
            
            for record_key, record_info in sorted(records.items(), key=lambda x: x[1].get("processed_at", "")):
                original = record_info.get("original_episode", "unknown")
                unique_name = record_info.get("unique_name", record_key)
                frame_start = record_info.get("frame_start", "?")
                frame_end = record_info.get("frame_end", "?")
                frame_count = record_info.get("frame_count", "?")
                processed_at = record_info.get("processed_at", "?")
                
                lines.append(f"\n唯一标识: {unique_name}")
                lines.append(f"  原始episode: {original}")
                lines.append(f"  帧范围: {frame_start} ~ {frame_end} (共 {frame_count} 帧)")
                lines.append(f"  处理时间: {processed_at}")
                lines.append("-" * 80)
            
            return "\n".join(lines)
        
        # 事件绑定
        refresh_btn.click(
            fn=refresh_episode_list,
            inputs=[data_dir, output_path],
            outputs=[episode_dd, episode_info]
        )
        
        refresh_records_btn.click(
            fn=refresh_processing_records,
            inputs=[data_dir, output_path],
            outputs=[processing_records]
        )
        
        def on_episode_change(data_dir_val, output_path_val, episode_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                              show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz, is_programmatic):
            """当选择 episode 时，更新信息并重置范围滑块，同时加载预览"""
            # 处理空值情况
            if not episode_val or not data_dir_val:
                return (
                    "请先选择 episode" if not episode_val else "所有 episode 已处理完成，请使用'批量创建 LeRobot 数据集'按钮创建数据集",
                    gr.update(maximum=0, value=0),
                    gr.update(maximum=0, value=0),
                    0,
                    gr.update(value=None),  # 使用gr.update避免Error显示
                    gr.update(value=None),  # 使用gr.update避免Error显示
                    "请先选择 episode" if not episode_val else "所有 episode 已处理完成，预览已清空。",
                    gr.update(value=None)  # 使用gr.update避免Error显示
                )
            
            # 如果是程序切换（处理完成后自动切换），快速返回，不加载预览
            if is_programmatic:
                try:
                    h5_path = pjoin(data_dir_val, episode_val, "data.hdf5")
                    if os.path.isfile(h5_path):
                        with h5py.File(h5_path, 'r') as f:
                            n = int(f['size'][()])  # 确保转换为 Python int
                            max_val = int(n - 1) if n > 0 else 0
                    else:
                        max_val = 0
                    info = f"Episode: {episode_val}\n总帧数: {max_val + 1}"
                    preview_info_text = f"起始帧: 0, 结束帧: {max_val}, 共 {max_val + 1} 帧\n点击'更新预览'按钮加载预览图像和轨迹图"
                    return (
                        info,
                        gr.update(maximum=int(max_val), value=0),
                        gr.update(maximum=int(max_val), value=int(max_val)),
                        0,
                        None,
                        None,
                        preview_info_text,
                        None
                    )
                except Exception as e:
                    return (
                        f"Episode: {episode_val}\n加载信息失败: {e}",
                        gr.update(maximum=0, value=0),
                        gr.update(maximum=0, value=0),
                        0,
                        None,
                        None,
                        f"加载失败: {e}",
                        None
                    )
            
            # 正常情况下的处理（用户手动选择episode）
            try:
                # 检查是否已处理
                is_processed, processed_key = is_episode_processed(output_path_val, episode_val, data_dir_val) if output_path_val else False
                info = get_episode_info(data_dir_val, episode_val)
                if is_processed and output_path_val:
                    record_info = load_processing_records(output_path_val, data_dir_val)[processed_key]
                    frame_start_old = record_info.get("frame_start", "?")
                    frame_end_old = record_info.get("frame_end", "?")
                    processed_at = record_info.get("processed_at", "?")
                    info += f"\n⚠️ 该episode已处理过！\n原处理: 帧{frame_start_old}~{frame_end_old}, 时间: {processed_at}\n重新处理将覆盖之前的记录。"
                
                ep_data = get_episode_data(data_dir_val, episode_val)
                n = int(ep_data["size"])  # 确保转换为 Python int
                
                if n <= 0:
                    return (
                        info,
                        gr.update(maximum=0, value=0),
                        gr.update(maximum=0, value=0),
                        0,
                        None,
                        None,
                        "该 episode 中没有数据",
                        None
                    )
                
                # 重置范围滑块为 [0, n-1]
                max_val = int(n - 1)
                
                # 加载起止帧预览（使用缓存加速）
                try:
                    img_start, _, info_start = preview_frame(data_dir_val, episode_val, 0, camera_side_val, use_cache=True)
                except Exception as e1:
                    img_start = None
                    info_start = f"加载起始帧失败: {e1}"
                
                try:
                    img_end, _, info_end = preview_frame(data_dir_val, episode_val, max_val, camera_side_val, use_cache=True)
                except Exception as e2:
                    img_end = None
                    info_end = f"加载结束帧失败: {e2}"
                
                preview_info_text = f"起始帧: 0, 结束帧: {max_val}, 共 {n} 帧\n起始帧: {info_start}\n结束帧: {info_end}"
                
                # 生成轨迹图（可能较慢，但必须执行）
                try:
                    traj_img = plot_trajectory(data_dir_val, episode_val, 0, max_val, arm_side_val, 
                                             float(pos_thresh_val), float(rot_thresh_val),
                                             show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                except Exception as e:
                    traj_img = None
                    preview_info_text += f"\n轨迹图生成失败: {e}"
                
                return (
                    info,
                    gr.update(maximum=int(max_val), value=0),
                    gr.update(maximum=int(max_val), value=int(max_val)),
                    0,
                    img_start,
                    img_end,
                    preview_info_text,
                    traj_img
                )
            except Exception as e:
                import traceback
                error_msg = f"加载失败: {e}\n{traceback.format_exc()}"
                return (
                    error_msg,
                    gr.update(maximum=0, value=0),
                    gr.update(maximum=0, value=0),
                    0,
                    None,
                    None,
                    f"预览加载失败: {e}",
                    None
                )
        
        episode_dd.change(
            fn=on_episode_change,
            inputs=[data_dir, output_path, episode_dd, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection, is_programmatic_switch],
            outputs=[episode_info, frame_start, frame_end, current_preview_frame, preview_img_start, preview_img_end, preview_info, trajectory_plot]
        )
        
        def on_start_frame_change(data_dir_val, episode_val, start_val, camera_side_val):
            """当起始帧滑块改变时，只更新起始帧预览"""
            if not episode_val or not data_dir_val:
                return None
            
            try:
                start_frame = int(start_val)
                img_start, _, _ = preview_frame(data_dir_val, episode_val, start_frame, camera_side_val)
                return img_start
            except Exception as e:
                return None
        
        def on_end_frame_change(data_dir_val, episode_val, end_val, camera_side_val):
            """当结束帧滑块改变时，只更新结束帧预览"""
            if not episode_val or not data_dir_val:
                return None
            
            try:
                end_frame = int(end_val)
                img_end, _, _ = preview_frame(data_dir_val, episode_val, end_frame, camera_side_val)
                return img_end
            except Exception as e:
                return None
        
        def on_frame_release(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                             show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """松开滑块时更新预览信息和轨迹图"""
            if not episode_val or not data_dir_val:
                return "请先选择 episode", None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                # 获取详细信息
                _, _, info_start = preview_frame(data_dir_val, episode_val, start_frame, camera_side_val)
                _, _, info_end = preview_frame(data_dir_val, episode_val, end_frame, camera_side_val)
                
                info = f"起始帧: {start_frame}, 结束帧: {end_frame}, 共 {end_frame - start_frame + 1} 帧\n起始帧: {info_start}\n结束帧: {info_end}"
                
                # 更新轨迹图
                traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                         float(pos_thresh_val), float(rot_thresh_val),
                                         show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                
                return info, traj_img
            except Exception as e:
                import traceback
                return f"预览失败: {e}\n{traceback.format_exc()}", None
        
        def on_camera_side_change(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                  show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """当相机侧改变时，更新所有预览"""
            if not episode_val or not data_dir_val:
                return None, None, "请先选择 episode", None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                img_start, _, info_start = preview_frame(data_dir_val, episode_val, start_frame, camera_side_val)
                img_end, _, info_end = preview_frame(data_dir_val, episode_val, end_frame, camera_side_val)
                
                info = f"起始帧: {start_frame}, 结束帧: {end_frame}, 共 {end_frame - start_frame + 1} 帧\n起始帧: {info_start}\n结束帧: {info_end}"
                
                # 更新轨迹图
                traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                         float(pos_thresh_val), float(rot_thresh_val),
                                         show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                
                return img_start, img_end, info, traj_img
            except Exception as e:
                import traceback
                return None, None, f"预览失败: {e}\n{traceback.format_exc()}", None
        
        def on_arm_side_change(data_dir_val, episode_val, start_val, end_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                               show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """当机械臂侧改变时，只更新轨迹图"""
            if not episode_val or not data_dir_val:
                return None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                         float(pos_thresh_val), float(rot_thresh_val),
                                         show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                return traj_img
            except Exception as e:
                return None
        
        def on_threshold_change(data_dir_val, episode_val, start_val, end_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                               show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """当阈值改变时，只更新轨迹图"""
            if not episode_val or not data_dir_val:
                return None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                         float(pos_thresh_val), float(rot_thresh_val),
                                         show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                return traj_img
            except Exception as e:
                return None
        
        def on_plot_option_change(data_dir_val, episode_val, start_val, end_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                  show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """当可视化选项改变时，只更新轨迹图"""
            if not episode_val or not data_dir_val:
                return None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                         float(pos_thresh_val), float(rot_thresh_val),
                                         show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                return traj_img
            except Exception as e:
                return None
        
        # 当起始帧滑块改变时（拖动时只更新起始帧预览，松开时更新预览信息）
        # 尝试使用input事件（如果Gradio版本支持）
        use_input_event = hasattr(frame_start, 'input')
        
        if use_input_event:
            # 拖动时只更新起始帧预览
            frame_start.input(
                fn=on_start_frame_change,
                inputs=[data_dir, episode_dd, frame_start, camera_side],
                outputs=[preview_img_start],
            )
            # 松开时只更新预览信息
            frame_start.change(
                fn=on_frame_release,
                inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                       show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                       show_xy_projection, show_xz_projection, show_yz_projection],
                outputs=[preview_info, trajectory_plot],
            )
        else:
            # 如果不支持input事件，change事件同时更新预览和信息
            def on_start_frame_change_with_info(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                                show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
                img = on_start_frame_change(data_dir_val, episode_val, start_val, camera_side_val)
                info, traj_img = on_frame_release(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                                  show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                return img, info, traj_img
            
            frame_start.change(
                fn=on_start_frame_change_with_info,
                inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                       show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                       show_xy_projection, show_xz_projection, show_yz_projection],
                outputs=[preview_img_start, preview_info, trajectory_plot],
            )
        
        # 当结束帧滑块改变时（拖动时只更新结束帧预览，松开时更新预览信息）
        if use_input_event:
            # 拖动时只更新结束帧预览
            frame_end.input(
                fn=on_end_frame_change,
                inputs=[data_dir, episode_dd, frame_end, camera_side],
                outputs=[preview_img_end],
            )
            # 松开时只更新预览信息
            frame_end.change(
                fn=on_frame_release,
                inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                       show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                       show_xy_projection, show_xz_projection, show_yz_projection],
                outputs=[preview_info, trajectory_plot],
            )
        else:
            # 如果不支持input事件，change事件同时更新预览和信息
            def on_end_frame_change_with_info(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                              show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
                img = on_end_frame_change(data_dir_val, episode_val, end_val, camera_side_val)
                info, traj_img = on_frame_release(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                                  show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                return img, info, traj_img
            
            frame_end.change(
                fn=on_end_frame_change_with_info,
                inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                       show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                       show_xy_projection, show_xz_projection, show_yz_projection],
                outputs=[preview_img_end, preview_info, trajectory_plot],
            )
        
        # 当相机侧改变时，更新所有预览
        camera_side.change(
            fn=on_camera_side_change,
            inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection],
            outputs=[preview_img_start, preview_img_end, preview_info, trajectory_plot],
        )
        
        # 当机械臂侧改变时，只更新轨迹图
        arm_side.change(
            fn=on_arm_side_change,
            inputs=[data_dir, episode_dd, frame_start, frame_end, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection],
            outputs=[trajectory_plot],
        )
        
        # 当阈值改变时，只更新轨迹图
        pos_outlier_threshold.change(
            fn=on_threshold_change,
            inputs=[data_dir, episode_dd, frame_start, frame_end, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection],
            outputs=[trajectory_plot],
        )
        rot_outlier_threshold.change(
            fn=on_threshold_change,
            inputs=[data_dir, episode_dd, frame_start, frame_end, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection],
            outputs=[trajectory_plot],
        )
        
        # 当可视化选项改变时，只更新轨迹图
        for checkbox in [show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                         show_xy_projection, show_xz_projection, show_yz_projection]:
            checkbox.change(
                fn=on_plot_option_change,
                inputs=[data_dir, episode_dd, frame_start, frame_end, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                       show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                       show_xy_projection, show_xz_projection, show_yz_projection],
                outputs=[trajectory_plot],
            )
        
        # 手动更新预览按钮（优化：添加异常处理，避免卡住）
        def on_update_preview_safe(data_dir_val, episode_val, start_val, end_val, camera_side_val, arm_side_val, pos_thresh_val, rot_thresh_val,
                                   show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz):
            """安全地更新预览，避免卡住"""
            if not episode_val or not data_dir_val:
                return None, None, "请先选择 episode", None
            
            try:
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 确保 end >= start
                if end_frame < start_frame:
                    end_frame = start_frame
                
                # 加载预览图像（使用缓存）
                try:
                    img_start, _, info_start = preview_frame(data_dir_val, episode_val, start_frame, camera_side_val, use_cache=True)
                except Exception as e:
                    img_start = None
                    info_start = f"加载起始帧失败: {e}"
                
                try:
                    img_end, _, info_end = preview_frame(data_dir_val, episode_val, end_frame, camera_side_val, use_cache=True)
                except Exception as e:
                    img_end = None
                    info_end = f"加载结束帧失败: {e}"
                
                info = f"起始帧: {start_frame}, 结束帧: {end_frame}, 共 {end_frame - start_frame + 1} 帧\n起始帧: {info_start}\n结束帧: {info_end}"
                
                # 生成轨迹图（可能较慢，添加超时保护）
                try:
                    traj_img = plot_trajectory(data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                                             float(pos_thresh_val), float(rot_thresh_val),
                                             show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz)
                except Exception as e:
                    traj_img = None
                    info += f"\n轨迹图生成失败: {e}"
                
                return img_start, img_end, info, traj_img
            except Exception as e:
                import traceback
                return None, None, f"更新预览失败: {e}\n{traceback.format_exc()}", None
        
        update_preview_btn.click(
            fn=on_update_preview_safe,
            inputs=[data_dir, episode_dd, frame_start, frame_end, camera_side, arm_side, pos_outlier_threshold, rot_outlier_threshold,
                   show_position, show_rotation, show_gripper, show_pos_velocity, show_rot_velocity,
                   show_xy_projection, show_xz_projection, show_yz_projection],
            outputs=[preview_img_start, preview_img_end, preview_info, trajectory_plot],
        )
        
        def on_process(
            data_dir_val,
            output_path_val,
            repo_name_val,
            episode_val,
            start_val,
            end_val,
            arm_side_val,
            current_log,
            pos_thresh_val,
            rot_thresh_val,
            show_pos,
            show_rot,
            show_grip,
            show_pos_vel,
            show_rot_vel,
            show_xy,
            show_xz,
            show_yz,
            camera_side_val,
            save_trajectory_image_val,
            trajectory_image_path_val
        ):

            # 清空缓存
            EPISODE_CACHE.clear()
            PREVIEW_CACHE.clear()
        
            """处理当前 episode，保存 data_cut.hdf5 并更新处理记录"""
            try:
                if not episode_val:
                    records_text = refresh_processing_records(data_dir_val, output_path_val) if data_dir_val and output_path_val else "无法加载处理记录"
                    return (
                        current_log + "\n请先选择 episode" if current_log else "请先选择 episode",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        None,
                        records_text,
                        False  # 不是程序切换
                    )
                
                # 从滑块获取起止帧
                start_frame = int(start_val)
                end_frame = int(end_val)
                
                # 如果episode已处理，先删除旧记录以便覆盖
                is_processed, processed_key = is_episode_processed(output_path_val, episode_val, data_dir_val) if output_path_val else False
                if is_processed:
                    print("alert! episode already processed!")
                    # 中断
                    raise Exception("episode already processed!")
                
                # 处理当前 episode（只保存 data_cut.hdf5 和更新记录，不创建 dataset）
                log_msg, unique_name = gen_cut_hdf5(
                    data_dir_val,
                    episode_val,
                    output_path_val,
                    start_frame,
                    end_frame,
                    update_records=True
                )
                
                # 追加日志而不是替换
                new_log = (current_log + "\n\n" + log_msg) if current_log else log_msg
                
                # 生成并保存轨迹可视化图（处理成功后保存，根据开关决定）
                if save_trajectory_image_val:
                    try:
                        traj_img = plot_trajectory(
                            data_dir_val, episode_val, start_frame, end_frame, arm_side_val,
                            float(pos_thresh_val), float(rot_thresh_val),
                            show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz
                        )
                        if traj_img is not None:
                            # 提取episode索引（从"episode0"中提取"0"）
                            import re
                            episode_match = re.search(r'episode(\d+)', episode_val)
                            episode_index = episode_match.group(1) if episode_match else episode_val
                            
                            # 确定保存路径
                            if trajectory_image_path_val and trajectory_image_path_val.strip():
                                # 使用用户指定的路径
                                save_dir = trajectory_image_path_val.strip()
                                # 如果是相对路径，相对于output_path
                                if not os.path.isabs(save_dir):
                                    save_dir = pjoin(output_path_val, save_dir)
                            else:
                                # 使用默认路径（output_path）
                                save_dir = output_path_val
                            
                            # 创建目录
                            os.makedirs(save_dir, exist_ok=True)
                            
                            # 保存图像
                            image_filename = f"{episode_index}_{start_frame}_{end_frame}_trajectory.png"
                            image_path = pjoin(save_dir, image_filename)
                            cv2.imwrite(image_path, cv2.cvtColor(traj_img, cv2.COLOR_RGB2BGR))
                            new_log = new_log + f"\n已保存轨迹可视化图: {image_path}"
                    except Exception as e:
                        import traceback
                        error_msg = f"保存轨迹可视化图失败: {e}\n{traceback.format_exc()}"
                        new_log = new_log + f"\n{error_msg}"
                
                # 确保 JSON 已保存，然后立即刷新处理记录显示
                # 强制刷新文件系统缓存并重新加载
                import time
                time.sleep(0.2)  # 短暂等待确保文件写入完成
                records_text = refresh_processing_records(data_dir_val, output_path_val)
                
                # 获取所有 episode，找到下一个未处理的
                all_episodes = list_episodes(data_dir_val)
                records = load_processing_records(output_path_val, data_dir_val)
                processed_episodes = {record_info.get("original_episode") for record_info in records.values()}
                unprocessed_episodes = [ep for ep in all_episodes if ep not in processed_episodes]
                
                # 初始化next_episode
                next_episode = None
                
                if len(unprocessed_episodes) > 0:
                    try:
                        # 如果当前episode在未处理列表中，找到下一个未处理的
                        if episode_val in unprocessed_episodes:
                            current_idx = unprocessed_episodes.index(episode_val)
                            if current_idx < len(unprocessed_episodes) - 1:
                                next_episode = unprocessed_episodes[current_idx + 1]
                        else:
                            # 如果当前episode不在未处理列表中（可能刚被处理），找到它在所有episodes中的位置
                            if episode_val in all_episodes:
                                current_idx = all_episodes.index(episode_val)
                                # 从下一个位置开始查找未处理的episode
                                for ep in all_episodes[current_idx + 1:]:
                                    if ep not in processed_episodes:
                                        next_episode = ep
                                        break
                    except ValueError:
                        # 当前 episode 不在列表中
                        pass
                        
                if next_episode:
                    # 更新 episode 下拉框和相关信息，并自动加载预览
                    try:
                        # 快速获取基本信息，不加载完整 episode 数据
                        h5_path = pjoin(data_dir_val, next_episode, "data.hdf5")
                        if os.path.isfile(h5_path):
                            with h5py.File(h5_path, 'r') as f:
                                n = int(f['size'][()])  # 确保转换为 Python int
                                max_val = int(n - 1) if n > 0 else 0
                        else:
                            max_val = 0
                        
                        info = f"Episode: {next_episode}\n总帧数: {max_val + 1}"
                        
                        # 自动加载预览图像和轨迹图
                        try:
                            img_start, _, info_start = preview_frame(data_dir_val, next_episode, 0, camera_side_val, use_cache=True)
                        except Exception as e1:
                            img_start = None
                            info_start = f"加载起始帧失败: {e1}"
                        
                        try:
                            img_end, _, info_end = preview_frame(data_dir_val, next_episode, max_val, camera_side_val, use_cache=True)
                        except Exception as e2:
                            img_end = None
                            info_end = f"加载结束帧失败: {e2}"
                        
                        preview_info_text = f"起始帧: 0, 结束帧: {max_val}, 共 {max_val + 1} 帧\n起始帧: {info_start}\n结束帧: {info_end}"
                        
                        # 生成轨迹图
                        try:
                            traj_img = plot_trajectory(
                                data_dir_val, next_episode, 0, max_val, arm_side_val,
                                float(pos_thresh_val), float(rot_thresh_val),
                                show_pos, show_rot, show_grip, show_pos_vel, show_rot_vel, show_xy, show_xz, show_yz
                            )
                        except Exception as e:
                            traj_img = None
                            preview_info_text += f"\n轨迹图生成失败: {e}"
                        
                        return (
                            new_log + f"\n\n已切换到下一个未处理的 episode: {next_episode}，预览已自动更新",
                            gr.update(value=next_episode),
                            info,
                            gr.update(maximum=int(max_val), value=0),
                            gr.update(maximum=int(max_val), value=int(max_val)),
                            0,
                            img_start,  # 自动加载预览图像
                            img_end,  # 自动加载预览图像
                            preview_info_text,
                            traj_img,  # 自动生成轨迹图
                            records_text,
                            True  # 设置程序切换标志
                        )
                    except Exception as e:
                        import traceback
                        return (
                            new_log + f"\n\n已切换到下一个 episode: {next_episode}，但加载信息失败: {e}\n{traceback.format_exc()}",
                            gr.update(value=next_episode),
                            f"加载失败: {e}",
                            gr.update(maximum=0, value=0),
                            gr.update(maximum=0, value=0),
                            0,
                            None,
                            None,
                            f"加载预览失败: {e}",
                            None,
                            records_text,
                            True  # 设置程序切换标志
                        )
                else:
                    # 所有episode处理完成
                    # 刷新episode列表，这样会清空episode_dd并触发on_episode_change显示友好提示
                    episode_update, episode_info_text = refresh_episode_list(data_dir_val, output_path_val, filter_processed=True)
                    completion_info = "✅ 所有 episode 已处理完成！\n\n提示: 使用'批量创建 LeRobot 数据集'按钮根据处理记录创建数据集"
                    return (
                        new_log + "\n\n所有 episode 已处理完成！\n提示: 使用'批量创建 LeRobot 数据集'按钮根据处理记录创建数据集",
                        episode_update,  # 更新episode_dd为空，触发on_episode_change
                        completion_info,  # episode_info 显示完成信息
                        gr.update(maximum=0, value=0),  # frame_start 重置
                        gr.update(maximum=0, value=0),  # frame_end 重置
                        0,  # current_preview_frame 重置为0
                        gr.update(value=None),  # preview_img_start 清空（使用gr.update避免Error）
                        gr.update(value=None),  # preview_img_end 清空（使用gr.update避免Error）
                        "所有 episode 已处理完成，预览已清空。\n请使用'批量创建 LeRobot 数据集'按钮创建数据集。",  # preview_info 显示友好提示
                        gr.update(value=None),  # trajectory_plot 清空（使用gr.update避免Error）
                        records_text,
                        False  # 不是程序切换
                    )

                
                # 如果没有下一个 episode，保持当前状态
                return (
                    new_log,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    None,
                    records_text,
                    False  # 不是程序切换
                )
                
            except Exception as e:
                import traceback
                error_msg = f"处理失败: {e}\n{traceback.format_exc()}"
                new_log = (current_log + "\n\n" + error_msg) if current_log else error_msg
                records_text = refresh_processing_records(data_dir_val, output_path_val) if data_dir_val and output_path_val else "无法加载处理记录"
                return (
                    new_log,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    None,
                    records_text,
                    False  # 不是程序切换
                )
        
        def on_abort_episode(data_dir_val, output_path_val, episode_val, current_log):
            """放弃当前episode"""
            try:
                if not episode_val:
                    return (current_log + "\n\n请先选择 episode") if current_log else "请先选择 episode"
                
                log_msg = abort_episode(data_dir_val, episode_val, output_path_val)
                new_log = (current_log + "\n\n" + log_msg) if current_log else log_msg
                
                # 刷新episode列表
                episode_update, episode_info_text = refresh_episode_list(data_dir_val, output_path_val, filter_processed=True)
                
                return new_log, episode_update, episode_info_text
            except Exception as e:
                import traceback
                error_msg = f"放弃操作失败: {e}\n{traceback.format_exc()}"
                new_log = (current_log + "\n\n" + error_msg) if current_log else error_msg
                episode_update, episode_info_text = refresh_episode_list(data_dir_val, output_path_val, filter_processed=True)
                return new_log, episode_update, episode_info_text
        
        def on_batch_create(data_dir_val, output_path_val, repo_name_val, current_log, progress=gr.Progress()):
            """批量创建lerobot数据集（带进度条）"""
            try:
                log_msg = create_lerobot_dataset_from_records(
                    data_dir_val,
                    output_path_val,
                    repo_name_val,
                    progress=progress
                )
                new_log = (current_log + "\n\n" + log_msg) if current_log else log_msg
                return new_log
            except Exception as e:
                import traceback
                error_msg = f"批量创建失败: {e}\n{traceback.format_exc()}"
                new_log = (current_log + "\n\n" + error_msg) if current_log else error_msg
                return new_log
        
        def on_process_with_records(
            data_dir_val,
            output_path_val,
            repo_name_val,
            episode_val,
            start_val,
            end_val,
            arm_side_val,
            current_log,
            pos_thresh_val,
            rot_thresh_val,
            show_pos,
            show_rot,
            show_grip,
            show_pos_vel,
            show_rot_vel,
            show_xy,
            show_xz,
            show_yz,
            camera_side_val,
            save_trajectory_image_val,
            trajectory_image_path_val
        ):
            """处理当前 episode，并更新处理记录显示"""
            # on_process已经返回了records_text，直接返回
            return on_process(
                data_dir_val,
                output_path_val,
                repo_name_val,
                episode_val,
                start_val,
                end_val,
                arm_side_val,
                current_log,
                pos_thresh_val,
                rot_thresh_val,
                show_pos,
                show_rot,
                show_grip,
                show_pos_vel,
                show_rot_vel,
                show_xy,
                show_xz,
                show_yz,
                camera_side_val,
                save_trajectory_image_val,
                trajectory_image_path_val
            )
        
        def on_batch_create_with_records(data_dir_val, output_path_val, repo_name_val, current_log, progress=gr.Progress()):
            """批量创建lerobot数据集，并更新处理记录显示"""
            result = on_batch_create(data_dir_val, output_path_val, repo_name_val, current_log, progress=progress)
            # 更新处理记录显示
            records_text = refresh_processing_records(data_dir_val, output_path_val)
            return result, records_text
        
        process_btn.click(
            fn=on_process_with_records,
            inputs=[
                data_dir,
                output_path,
                repo_name,
                episode_dd,
                frame_start,
                frame_end,
                arm_side,
                process_log,
                pos_outlier_threshold,
                rot_outlier_threshold,
                show_position,
                show_rotation,
                show_gripper,
                show_pos_velocity,
                show_rot_velocity,
                show_xy_projection,
                show_xz_projection,
                show_yz_projection,
                camera_side,
                save_trajectory_image,
                trajectory_image_path
            ],
            outputs=[
                process_log,
                episode_dd,
                episode_info,
                frame_start,
                frame_end,
                current_preview_frame,
                preview_img_start,
                preview_img_end,
                preview_info,
                trajectory_plot,
                processing_records,
                is_programmatic_switch
            ]
        )
        
        abort_btn.click(
            fn=on_abort_episode,
            inputs=[data_dir, output_path, episode_dd, process_log],
            outputs=[process_log, episode_dd, episode_info]
        )
        
        batch_create_btn.click(
            fn=on_batch_create_with_records,
            inputs=[data_dir, output_path, repo_name, process_log],
            outputs=[process_log, processing_records]
        )
    
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7850)

