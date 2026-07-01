"""Image and depth preprocessing matching the training pipeline."""

import os
import tempfile
from typing import Optional

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 640  # Training resolution: longest edge = 640


def resize_to_target(
    image: np.ndarray, target_size: int = TARGET_SIZE, interp=cv2.INTER_LINEAR
) -> tuple[np.ndarray, float]:
    """Resize so longest edge equals target_size. Returns (resized, scale_factor)."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized, scale


def prepare_inputs(
    rgb: Image.Image,
    depth: np.ndarray,
    tmp_dir: Optional[str] = None,
) -> dict:
    """Prepare model inputs matching the training data format.

    Args:
        rgb: PIL RGB image
        depth: Depth map as float32 numpy array (H, W) in meters
        tmp_dir: Temporary directory for saving intermediate files.
                 If None, creates one automatically.

    Returns:
        dict with keys:
            image_path: path to resized RGB png
            npz_path: path to pcd .npz file
            rgb_resized: resized RGB as numpy (H, W, 3)
            depth_resized: resized depth as numpy (H, W)
            scale: scale factor applied
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="hamster3d_")

    rgb_np = np.array(rgb.convert("RGB"))
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    # Resize both to target
    rgb_resized, scale = resize_to_target(rgb_np, TARGET_SIZE, cv2.INTER_LINEAR)
    depth_resized, _ = resize_to_target(depth, TARGET_SIZE, cv2.INTER_NEAREST)

    new_h, new_w = rgb_resized.shape[:2]

    # Build pcd array (H, W, 4) = [x, y, z, mask]
    mask = ((depth_resized > 0.01) & (depth_resized < 10.0)).astype(np.float32)
    pcd = np.zeros((new_h, new_w, 4), dtype=np.float32)
    pcd[:, :, 2] = depth_resized
    pcd[:, :, 3] = mask

    # Save files
    img_path = os.path.join(tmp_dir, "frame_0_640.png")
    npz_path = os.path.join(tmp_dir, "frame_0_640.npz")
    cv2.imwrite(img_path, cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
    np.savez_compressed(npz_path, pcd=pcd.astype(np.float16))

    return {
        "image_path": img_path,
        "npz_path": npz_path,
        "rgb_resized": rgb_resized,
        "depth_resized": depth_resized,
        "scale": scale,
    }


def build_geometry_inputs(
    rgb_resized: np.ndarray,
    depth_resized: np.ndarray,
    device: str = "cuda",
) -> dict:
    """Build geometry_encoder_inputs and depth_maps tensors from numpy arrays.

    Args:
        rgb_resized: RGB array (H, W, 3), uint8 [0, 255]
        depth_resized: Depth array (H, W), float32 in meters

    Returns:
        dict with geometry_encoder_inputs and depth_maps (as lists of tensors)
    """
    import torch

    # RGB: [H, W, 3] uint8 -> [1, 3, H, W] float32 [0, 1]
    rgb_tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Depth: [H, W] -> [1, H, W]. Round-trip through float16 to match the training /
    # eval / Gradio pipeline (depth is stored in a float16 .npz there); feeding raw
    # float32 here drifts the geometry encoder and the greedy decode.
    depth_resized = np.asarray(depth_resized).astype(np.float16).astype(np.float32)
    depth_tensor = torch.from_numpy(depth_resized).float().unsqueeze(0)

    rgb_tensor = rgb_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    return {
        "geometry_encoder_inputs": [rgb_tensor],
        "depth_maps": [depth_tensor],
    }


# v5 prompt format
V5_SYSTEM_PROMPT = ""
V5_HUMAN_PROMPT_SUFFIX = (
    "Predict the full manipulation trajectory as point_3d waypoints "
    "with depth and gripper state in JSON."
)


def build_v5_messages(instruction: str) -> list[dict]:
    """Build Qwen3-VL chat messages in v5 format."""
    human_msg = f"{instruction}\n{V5_HUMAN_PROMPT_SUFFIX}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": human_msg},
            ],
        }
    ]
