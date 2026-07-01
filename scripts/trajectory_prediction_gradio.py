#!/usr/bin/env python3
"""
3D Trajectory Prediction Gradio App for Qwen3-VL Geometry Model

Predicts 2.5D robot manipulation waypoints from RGB + metric depth.
Visualizes predicted trajectories in 2D (overlaid on image) and 3D (interactive Plotly).

Input:
    - RGB image (any resolution — auto-resized to longest-edge=640 to match training)
    - Metric depth map (.npy, float32, meters)

Usage:
    conda activate 3d_hamster
    CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_prediction_gradio.py
    CUDA_VISIBLE_DEVICES=0 python scripts/trajectory_prediction_gradio.py --autoload
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image

# Setup paths so the self-contained hamster3d package is importable
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default merged model directory (no LLaMA-Factory / LoRA adapter needed)
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "ckpt")

# Training resolution: images resized so longest edge = 640
TARGET_SIZE = 640

# v5 inference format (matches infer_droid_spatial_bench.py).
# System prompt is EMPTY; the task instruction + a format-request suffix go in the user turn,
# and the model replies with a JSON array of point_2d/point_3d waypoints.
V5_SYSTEM_PROMPT = ""

# General-VQA style: free-form question, no trajectory suffix, output is plain text.
VQA_STYLE = "General VQA"

# 2D bounding-box style: locate an object, output [{"bbox_2d":[x1,y1,x2,y2],"label":...}] (0-1000).
BBOX_STYLE = "2D Bounding Box"

# Canonical user-turn suffixes (one per prompt style), copied verbatim from
# infer_droid_spatial_bench.py so inference matches the benchmark exactly.
# A None suffix means free-form (general VQA): the instruction is sent as-is.
V5_PROMPT_SUFFIXES = {
    # metric 3D end-effector trajectory (point_3d with depth)
    "3D Trajectory": (
        "Predict the full manipulation trajectory as point_3d waypoints "
        "with depth and gripper state in JSON."
    ),
    # 2D pixel trajectory (point_2d, no depth)
    "2D Trajectory": (
        "Predict the full manipulation trajectory as point_2d waypoints "
        "with gripper state in JSON."
    ),
    # RefSpatial-style 3D pointing — [u, v, depth] points (u,v 0-1000, depth in meters)
    "3D Pointing": "Report the point_3d location in JSON.",
    # RoboPoint-style 2D pointing — independent [u, v] points (0-1000), with labels
    "2D Pointing": "Report point_2d locations in JSON.",
    # object localization (dedicated bbox template below; value is a placeholder)
    BBOX_STYLE: None,
    # free-form question answering — no suffix, output is plain text (not a trajectory)
    VQA_STYLE: None,
}
V5_PROMPT_STYLES = list(V5_PROMPT_SUFFIXES.keys())
V5_DEFAULT_STYLE = "3D Trajectory"

# Pointing styles output independent points (not a connected trajectory) — drawn as
# numbered dots rather than a path.
POINTING_STYLES = {"2D Pointing", "3D Pointing"}

# Backward-compat alias (some helpers still reference SYSTEM_PROMPT as the system value)
SYSTEM_PROMPT = V5_SYSTEM_PROMPT


def build_v5_human_message(instruction: str, prompt_style: str) -> str:
    """Build the user-turn message: task instruction + format-request suffix.

    - bbox (2D): the instruction is the object to locate, wrapped in the lvis_bbox2d template.
    - general VQA (suffix is None): the instruction is returned verbatim.
    - otherwise: instruction + "\\n" + the style's format-request suffix.
    """
    instr = instruction.strip()
    if prompt_style == BBOX_STYLE:
        return f"I'm looking for {instr} in this image. Can you locate it? Report bbox coordinates in JSON format."
    if prompt_style not in V5_PROMPT_SUFFIXES:
        prompt_style = V5_DEFAULT_STYLE
    suffix = V5_PROMPT_SUFFIXES[prompt_style]
    if suffix is None:
        return instr
    return f"{instr}\n{suffix}"


# ── Image/Depth Preprocessing (matches training pipeline) ───────────────────


def resize_to_target(image: np.ndarray, target_size: int = TARGET_SIZE, interp=cv2.INTER_LINEAR):
    """Resize so longest edge equals target_size. Returns (resized, scale_factor)."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized, scale


def prepare_inputs_from_arrays(rgb: np.ndarray, depth: np.ndarray, tmp_dir: str):
    """
    Prepare model inputs from raw RGB + metric depth arrays (matches training format).

    1. Resize RGB so longest edge = 640 → save as frame_0_640.png
    2. Resize depth (nearest) to same shape → build pcd (H, W, 4) → save as frame_0_640.npz
    3. The plugin auto-discovers .npz from .png path (same stem), or we pass it explicitly.

    Returns:
        (image_path, npz_path, rgb_resized, depth_resized, scale)
    """
    rgb_resized, scale = resize_to_target(rgb, TARGET_SIZE, cv2.INTER_LINEAR)
    depth_resized, _ = resize_to_target(depth, TARGET_SIZE, cv2.INTER_NEAREST)

    new_h, new_w = rgb_resized.shape[:2]

    # Build pcd array (H, W, 4) = [x, y, z, mask]
    # The model only reads z (depth) and mask from this; x/y are unused by the depth encoder.
    mask = ((depth_resized > 0.01) & (depth_resized < 10.0)).astype(np.float32)
    pcd = np.zeros((new_h, new_w, 4), dtype=np.float32)
    pcd[:, :, 2] = depth_resized  # z = depth
    pcd[:, :, 3] = mask

    # Save with matched names so auto-discovery works too
    img_path = os.path.join(tmp_dir, "frame_0_640.png")
    npz_path = os.path.join(tmp_dir, "frame_0_640.npz")
    cv2.imwrite(img_path, cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
    np.savez_compressed(npz_path, pcd=pcd.astype(np.float16))

    return img_path, npz_path, rgb_resized, depth_resized, scale


def prepare_inputs(rgb_pil: Image.Image, depth_npy_path: str, tmp_dir: str):
    """Prepare model inputs from a PIL RGB image + a metric depth .npy path."""
    rgb = np.array(rgb_pil.convert("RGB"))
    depth = np.load(depth_npy_path).astype(np.float32)
    img_path, npz_path, rgb_resized, depth_resized, _ = prepare_inputs_from_arrays(rgb, depth, tmp_dir)
    return img_path, npz_path, rgb_resized, depth_resized


# ── Trajectory Parsing ───────────────────────────────────────────────────────


def parse_trajectory(output: str) -> tuple[list[list[float]], list[Optional[str]]]:
    """
    Parse waypoints and actions from model output.
    Expected: <ans>[[u, v, d], <action>Close Gripper</action>, ...]</ans>
    """
    waypoints, actions = [], []

    ans_match = re.search(r"<ans>(.*?)</ans>", output, re.DOTALL)
    if not ans_match:
        ans_match = re.search(r"\[\[.*?\]\]", output, re.DOTALL)
        if not ans_match:
            return [], []
        content = ans_match.group(0)
    else:
        content = ans_match.group(1)

    coord_pat = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
    action_pat = r"<action>(.*?)</action>"
    parts = re.split(action_pat, content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            for c in re.findall(coord_pat, part):
                waypoints.append([float(c[0]), float(c[1]), float(c[2])])
                actions.append(None)
        else:
            if actions:
                actions[-1] = part.strip()

    return waypoints, actions


# ── 2D Visualization ─────────────────────────────────────────────────────────

COLOR_WP = (0, 255, 0)
COLOR_GRASP = (255, 0, 0)
COLOR_RELEASE = (0, 0, 255)
COLOR_LINE = (255, 255, 0)


def visualize_2d(image: np.ndarray, waypoints, actions) -> np.ndarray:
    """Draw trajectory on image with numbered waypoints and depth labels."""
    if not waypoints:
        return image

    img = image.copy()
    h, w = img.shape[:2]
    pixels = [(int(wp[0] / 1000 * w), int(wp[1] / 1000 * h)) for wp in waypoints]

    for i in range(len(pixels) - 1):
        cv2.line(img, pixels[i], pixels[i + 1], COLOR_LINE, 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(pixels):
        act = actions[i] if i < len(actions) else None
        if act and "Close" in act:
            color, r = COLOR_GRASP, 12
        elif act and "Open" in act:
            color, r = COLOR_RELEASE, 12
        else:
            color, r = COLOR_WP, 8

        cv2.circle(img, (px, py), r, color, -1)
        cv2.circle(img, (px, py), r, (255, 255, 255), 2)
        cv2.putText(img, str(i), (px - 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, f"d={waypoints[i][2]:.2f}m", (px + 15, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Legend
    y = 30
    for label, color, xo in [("Waypoint", COLOR_WP, 10), ("Grasp", COLOR_GRASP, 110), ("Release", COLOR_RELEASE, 180)]:
        cv2.circle(img, (xo, y - 5), 6, color, -1)
        cv2.putText(img, label, (xo + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img


# ── BBox (2D) Parsing + Visualization ─────────────────────────────────────────


def visualize_points(image: np.ndarray, points) -> np.ndarray:
    """Draw independent numbered points (no connecting line) for pointing tasks.

    points: list of [u, v, ...] with u, v normalized to 0-1000.
    """
    if not points:
        return image
    img = image.copy()
    h, w = img.shape[:2]
    for i, p in enumerate(points):
        px, py = int(p[0] / 1000 * w), int(p[1] / 1000 * h)
        cv2.circle(img, (px, py), 8, (0, 255, 0), -1)
        cv2.circle(img, (px, py), 8, (255, 255, 255), 2)
        cv2.putText(img, str(i + 1), (px + 11, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def parse_bbox_2d(output: str) -> list[tuple[float, float, float, float, str]]:
    """Parse [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}] (coords 0-1000) from output."""
    m = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    raw = m.group(1) if m else output.strip()
    if not raw.lstrip().startswith("["):
        arr = re.search(r"\[.*\]", raw, re.DOTALL)
        if not arr:
            return []
        raw = arr.group(0)
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []

    boxes = []
    for e in entries:
        if not isinstance(e, dict) or "bbox_2d" not in e:
            continue
        b = e["bbox_2d"]
        if len(b) < 4:
            continue
        boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(e.get("label", ""))))
    return boxes


def visualize_bbox(image: np.ndarray, boxes) -> np.ndarray:
    """Draw 2D bounding boxes (coords normalized 0-1000) with labels."""
    if not boxes:
        return image
    img = image.copy()
    h, w = img.shape[:2]
    palette = [(0, 255, 0), (255, 80, 0), (0, 160, 255), (255, 0, 200), (255, 220, 0)]
    for i, (x1, y1, x2, y2, label) in enumerate(boxes):
        p1 = (int(x1 / 1000 * w), int(y1 / 1000 * h))
        p2 = (int(x2 / 1000 * w), int(y2 / 1000 * h))
        color = palette[i % len(palette)]
        cv2.rectangle(img, p1, p2, color, 2)
        tag = label or f"obj{i}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, tag, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


# ── 3D Visualization ─────────────────────────────────────────────────────────


def visualize_3d(waypoints, actions, image_h: int, image_w: int, connect: bool = True) -> go.Figure:
    """Create interactive 3D trajectory plot."""
    fig = go.Figure()
    if not waypoints:
        fig.update_layout(title="No waypoints to display")
        return fig

    # Simple back-projection (assume centered pinhole)
    fx = fy = float(max(image_w, image_h))
    cx, cy = image_w / 2.0, image_h / 2.0

    pts = []
    for wp in waypoints:
        px = wp[0] / 1000 * image_w
        py = wp[1] / 1000 * image_h
        d = wp[2]
        pts.append([-(px - cx) * d / fx, -(py - cy) * d / fy, d])
    pts = np.array(pts)

    if connect and len(pts) > 1:
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines", line=dict(color="yellow", width=6), name="Trajectory",
        ))

    colors, sizes = [], []
    for a in actions:
        if a and "Close" in a:
            colors.append("red"); sizes.append(14)
        elif a and "Open" in a:
            colors.append("blue"); sizes.append(14)
        else:
            colors.append("lime"); sizes.append(10)
    while len(colors) < len(pts):
        colors.append("lime"); sizes.append(10)

    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
        text=[str(i) for i in range(len(pts))],
        textposition="top center", textfont=dict(size=12, color="white"),
        name="Waypoints",
        hovertemplate="WP %{text}<br>X:%{x:.3f}<br>Y:%{y:.3f}<br>Z:%{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z (Depth)",
            aspectmode="data",
            camera=dict(eye=dict(x=0, y=0.3, z=-1.5), up=dict(x=0, y=1, z=0)),
            bgcolor="rgb(20, 20, 30)",
        ),
        title="3D Trajectory", showlegend=True, height=550,
        paper_bgcolor="rgb(30, 30, 40)", font=dict(color="white"),
    )
    return fig


# ── Depth Colorization ───────────────────────────────────────────────────────


def colorize_depth(depth: np.ndarray, min_d: float = 0.1, max_d: float = 3.0) -> np.ndarray:
    d = np.clip(depth, min_d, max_d)
    d = ((d - min_d) / (max_d - min_d) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[depth <= 0] = [0, 0, 0]
    return colored


# ── 3D Scene + Trajectory Visualization ──────────────────────────────────────


def _create_sphere_points(center, radius=0.005, n=150):
    phi = np.random.uniform(0, 2 * np.pi, n)
    ct = np.random.uniform(-1, 1, n)
    theta = np.arccos(ct)
    return np.stack([
        center[0] + radius * np.sin(theta) * np.cos(phi),
        center[1] + radius * np.sin(theta) * np.sin(phi),
        center[2] + radius * np.cos(theta),
    ], axis=1)


def _create_tube_points(p1, p2, radius=0.002):
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-8:
        return np.empty((0, 3))
    d /= L
    perp1 = np.cross(d, [1, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    pts = []
    for ti in np.linspace(0, 1, max(int(L / 0.002), 10)):
        c = p1 + ti * (p2 - p1)
        for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            pts.append(c + radius * (np.cos(a) * perp1 + np.sin(a) * np.cross(d, perp1)))
    return np.array(pts)


def _uvd_to_xyz(coords, intrinsics_3x3, img_w, img_h, uvd_norm=1000.0):
    """Convert [u, v, depth] (normalized 0-1000) to XYZ in camera frame."""
    K = np.array(intrinsics_3x3, dtype=np.float64)
    Kinv = np.linalg.inv(K)
    u_px = (coords[:, 0] / uvd_norm) * img_w
    v_px = (coords[:, 1] / uvd_norm) * img_h
    pixels = np.stack([u_px, v_px, np.ones(len(u_px))], axis=1)
    return coords[:, 2:3] * (pixels @ Kinv.T)


def build_scene_pcd(image_path, npz_path, intrinsics_3x3):
    """Build scene point cloud from RGB image + depth NPZ + camera intrinsics."""
    try:
        import open3d as o3d
    except ImportError:
        return None, None

    rgb = np.array(Image.open(image_path).convert("RGB"))
    npz = np.load(npz_path)
    pcd_data = npz["pcd"]  # (H, W, 4) — x, y, z=depth, mask
    depth = pcd_data[:, :, 2].astype(np.float32)

    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

    H, W = depth.shape[:2]
    K = np.array(intrinsics_3x3, dtype=np.float64)
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.set_intrinsics(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb.astype(np.uint8)),
        o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam, extrinsic=np.eye(4))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # Filter out near-camera noise
    mask = pts[:, 2] >= 0.1
    return pts[mask], colors[mask]


def default_intrinsics(img_h: int, img_w: int):
    """Assumed pinhole intrinsics for examples that ship no camera calibration:
    focal = max(H, W), centered principal point. Enough to render a geometrically
    reasonable scene point cloud aligned with the predicted trajectory."""
    f = float(max(img_h, img_w))
    return [[f, 0.0, img_w / 2.0], [0.0, f, img_h / 2.0], [0.0, 0.0, 1.0]]


def build_scene_pcd_simple(rgb: np.ndarray, depth: np.ndarray, intrinsics_3x3,
                           depth_trunc: float = 3.0, stride: int = 2):
    """Unproject an RGB-D frame to a colored point cloud with numpy (no open3d).
    Matches `_uvd_to_xyz`'s convention so the scene and trajectory align."""
    H, W = depth.shape[:2]
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    K = np.asarray(intrinsics_3x3, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[vs, us].astype(np.float32)
    valid = (z > 0.1) & (z < depth_trunc)
    z, us, vs = z[valid], us[valid], vs[valid]
    if z.size == 0:
        return None, None
    pts = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=1)
    cols = rgb[vs, us].astype(np.float32) / 255.0
    return pts, cols


def build_3d_scene_figure(
    scene_pts, scene_cols,
    traj_dict: dict[str, tuple[np.ndarray, list[float]]],
    title: str = "",
    subsample: int = 2,
):
    """Build a Plotly figure with scene point cloud + multiple trajectory overlays.

    Args:
        scene_pts: (N, 3) scene XYZ
        scene_cols: (N, 3) scene RGB [0-1]
        traj_dict: {"label": (xyz_array, [r,g,b])} for each trajectory
        title: plot title
        subsample: downsample scene points
    """
    fig = go.Figure()

    if scene_pts is not None and len(scene_pts) > 0:
        sp = scene_pts[::subsample]
        sr = (scene_cols[::subsample] * 255).astype(np.uint8)
        fig.add_trace(go.Scatter3d(
            x=sp[:, 0], y=sp[:, 1], z=sp[:, 2], mode="markers",
            marker=dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in sr], opacity=0.6),
            name="Scene", hoverinfo="skip",
        ))

    for label, (xyz, color) in traj_dict.items():
        if xyz is None or len(xyz) < 2:
            continue
        c = np.array(color)
        tpts, tcols = [], []
        # Start sphere (brighter)
        sp = _create_sphere_points(xyz[0], radius=0.008, n=400)
        tpts.append(sp)
        tcols.append(np.tile(np.clip(c * 1.3, 0, 1), (len(sp), 1)))
        # End sphere (darker)
        ep = _create_sphere_points(xyz[-1], radius=0.008, n=400)
        tpts.append(ep)
        tcols.append(np.tile(c * 0.7, (len(ep), 1)))
        # Tube segments
        for i in range(len(xyz) - 1):
            tube = _create_tube_points(xyz[i], xyz[i + 1], radius=0.003)
            if len(tube) > 0:
                tpts.append(tube)
                tcols.append(np.tile(c, (len(tube), 1)))

        tpts = np.vstack(tpts)
        tcols = np.vstack(tcols)
        tr_rgb = (np.clip(tcols, 0, 1) * 255).astype(np.uint8)
        fig.add_trace(go.Scatter3d(
            x=tpts[:, 0], y=tpts[:, 1], z=tpts[:, 2], mode="markers",
            marker=dict(size=2.5, color=[f"rgb({r},{g},{b})" for r, g, b in tr_rgb], opacity=1.0),
            name=label, hoverinfo="skip",
        ))

    fig.update_layout(
        title=title, height=650,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data", bgcolor="white",
            camera=dict(eye=dict(x=0, y=0, z=-1.5), up=dict(x=0, y=-1, z=0)),
        ),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ── GT Parsing (ShareGPT v5 structured JSON) ─────────────────────────────────


def parse_gt_structured_json(gpt_value: str) -> tuple[list[list[float]], list[Optional[str]]]:
    """Parse waypoints from v5 structured JSON output.

    Handles both formats:
      - structured_json_from_grasp:        [{"point_3d": [u, v, d], "gripper": "close"}, ...]
      - structured_json_2d_to_3d_from_grasp: [{"point_2d": [u, v], ...}, ..., {"point_3d": [u, v, d], ...}, ...]

    For the 2d_to_3d format the entries mix point_2d and point_3d. We prefer point_3d
    globally — if any entry carries point_3d we keep only those (the final 3D trajectory)
    and drop the intermediate 2D pass; otherwise we fall back to point_2d (depth = 0).
    """
    # Extract the JSON array: prefer a ```json fence, else the first [...] block.
    m = re.search(r"```json\s*(.*?)\s*```", gpt_value, re.DOTALL)
    raw = m.group(1) if m else gpt_value.strip()
    if not raw.lstrip().startswith("["):
        arr = re.search(r"\[.*\]", raw, re.DOTALL)
        if not arr:
            return [], []
        raw = arr.group(0)

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return [], []
    if not isinstance(entries, list):
        return [], []

    # Prefer point_3d globally when present (handles the 2d_to_3d mixed output).
    key = "point_3d" if any(isinstance(e, dict) and "point_3d" in e for e in entries) else "point_2d"

    waypoints, actions = [], []
    for entry in entries:
        if not isinstance(entry, dict) or key not in entry:
            continue
        pt = entry[key]
        waypoints.append([float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0])
        grip = entry.get("gripper", "none")
        if grip == "close":
            actions.append("Close Gripper")
        elif grip == "open":
            actions.append("Open Gripper")
        else:
            actions.append(None)
    return waypoints, actions


# ── Examples Browser (bundled examples/ folder) ──────────────────────────────

EXAMPLES_DIR = str(PROJECT_ROOT / "examples")


def scan_examples(examples_dir: str) -> list[dict]:
    """Scan an examples/ folder for `<name>_rgb.png` + `<name>_depth.npy`
    (+ optional `<name>_instruction.txt`, `<name>_camera.json`) bundles, like the
    Dataset Browser loads samples. `_camera.json` carries the real camera
    intrinsics (and extrinsics) so the 3D scene is metrically correct.
    """
    samples = []
    if not os.path.isdir(examples_dir):
        return samples
    for rgb_name in sorted(f for f in os.listdir(examples_dir) if f.endswith("_rgb.png")):
        prefix = rgb_name[: -len("_rgb.png")]
        rgb_path = os.path.join(examples_dir, rgb_name)
        depth_path = os.path.join(examples_dir, f"{prefix}_depth.npy")
        instr_path = os.path.join(examples_dir, f"{prefix}_instruction.txt")
        cam_path = os.path.join(examples_dir, f"{prefix}_camera.json")
        if not os.path.isfile(depth_path):
            continue
        instruction = ""
        if os.path.isfile(instr_path):
            with open(instr_path) as f:
                instruction = f.read().strip()
        intrinsics, extrinsics = None, None
        if os.path.isfile(cam_path):
            try:
                with open(cam_path) as f:
                    cam = json.load(f)
                intrinsics = cam.get("intrinsics")
                extrinsics = cam.get("extrinsics")
            except Exception:
                pass
        samples.append({
            "name": prefix,
            "rgb_path": rgb_path,
            "depth_npy_path": depth_path,
            "instruction": instruction,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        })
    return samples


class ExamplesBrowser:
    """Manages samples discovered in the bundled examples/ folder."""

    def __init__(self):
        self.samples: list[dict] = []
        self.current_idx: int = 0

    def load(self, examples_dir: str) -> str:
        self.samples = scan_examples(examples_dir)
        self.current_idx = 0
        if not self.samples:
            return f"No examples found in {examples_dir}"
        return f"Loaded {len(self.samples)} example(s) from {examples_dir}"

    def get(self, idx: int) -> Optional[dict]:
        if not self.samples:
            return None
        self.current_idx = max(0, min(len(self.samples) - 1, int(idx)))
        return self.samples[self.current_idx]


examples_browser = ExamplesBrowser()


# ── Conversation Formatting ──────────────────────────────────────────────────


def format_conversation(system: str, user: str, assistant: str) -> str:
    """Render the full system/user/assistant exchange as one readable block."""
    return (
        "════════ SYSTEM ════════\n"
        f"{system.strip()}\n\n"
        "════════ USER ════════\n"
        f"{user.strip()}\n\n"
        "════════ ASSISTANT ════════\n"
        f"{assistant.strip()}"
    )


# ── Model Server ─────────────────────────────────────────────────────────────


class ModelServer:
    """Self-contained merged-model server (no LLaMA-Factory dependency).

    Loads a merged Qwen3-VL-Geometry model via AutoModelForImageTextToText +
    AutoProcessor and runs greedy generation, returning the raw decoded text so
    the gradio callers' parse_* functions behave identically to the original.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_path: Optional[str] = None
        self.device = "cuda:0"

    def load(self, model_path: str) -> str:
        # Normalize the path: strip a trailing slash and resolve relative paths
        # against the repo root. This avoids HuggingFace treating e.g. "ckpt/" as
        # a hub repo id ("Repo id must use alphanumeric chars...") when the gradio
        # is launched from a different working directory.
        model_path = os.path.expanduser(str(model_path)).rstrip("/")
        if not os.path.isabs(model_path):
            cand = PROJECT_ROOT / model_path
            if cand.is_dir():
                model_path = str(cand)
        model_path = model_path or str(PROJECT_ROOT / "ckpt")

        if self.model is not None and self.loaded_path == model_path:
            return f"Already loaded: {os.path.basename(model_path)}"

        self.model = None
        self.processor = None
        self.loaded_path = None

        if not os.path.isdir(model_path):
            return f"Error: model not found — {model_path}"

        try:
            # Register the custom Qwen3-VL geometry model class
            from hamster3d.model import register_qwen3_vl_geometry
            try:
                register_qwen3_vl_geometry()
            except Exception:
                pass

            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=self.device,
            )
            self.model.eval()
            self.loaded_path = model_path
            return f"Loaded: {os.path.basename(model_path)}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_path: str, npz_path: str, query: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        if not self.is_loaded():
            return "Error: model not loaded"

        try:
            device = self.device

            # ── Geometry inputs (match hamster3d.inference.preprocessing.build_geometry_inputs) ──
            # RGB from the 640-longest-edge PNG the gradio already saved.
            rgb = np.array(Image.open(image_path).convert("RGB"))
            # Depth from the npz EXACTLY like the reference (reproduces the float16 round-trip).
            pcd = np.load(npz_path)["pcd"]  # (H, W, 4), saved float16
            depth = pcd[:, :, 2].astype(np.float32)

            rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)
            # Cast geometry/depth to the model dtype (bf16) so the encoder sees the
            # SAME precision as the llamafactory reference (which casts the batch to
            # model dtype before generate). Feeding float32 into a bf16 model changes
            # vision/geometry precision and drifts greedy argmax.
            mdtype = next(self.model.parameters()).dtype
            geometry_encoder_inputs = [rgb_tensor.to(device=device, dtype=mdtype)]
            depth_maps = [depth_tensor.to(device=device, dtype=mdtype)]

            # ── Chat prompt (match the reference token stream) ──
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            })
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.processor(
                text=[text],
                images=[Image.open(image_path).convert("RGB")],
                return_tensors="pt",
            ).to(device)
            # Cast floating inputs (pixel_values) to the model dtype (bf16) to match
            # the reference: pixel_values are bf16-identical, but float32 -> bf16 model
            # changes vision-tower precision and drifts greedy decoding.
            for _k, _v in list(model_inputs.items()):
                if torch.is_tensor(_v) and torch.is_floating_point(_v):
                    model_inputs[_k] = _v.to(mdtype)
            model_inputs["geometry_encoder_inputs"] = geometry_encoder_inputs
            model_inputs["depth_maps"] = depth_maps

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            input_len = model_inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
            raw_output = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return raw_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"


server = ModelServer()


# ── Gradio UI ────────────────────────────────────────────────────────────────


def build_ui(default_model_path: str) -> gr.Blocks:
    with gr.Blocks(
        title="3D Trajectory Prediction",
        theme=gr.themes.Soft(),
        css=".mono textarea { font-family: monospace; font-size: 13px; }",
    ) as demo:
        gr.Markdown(
            "# 3D Trajectory Prediction\n"
            "Qwen3-VL Geometry — predict 2.5D robot manipulation waypoints from **RGB + metric depth**.\n\n"
            f"Images are auto-resized to longest-edge={TARGET_SIZE} to match training resolution."
        )

        # ── Model ──
        with gr.Accordion("Model Settings", open=False):
            with gr.Row():
                model_path_box = gr.Textbox(label="Model Path", value=default_model_path, scale=4)
                load_btn = gr.Button("Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(label="Status", interactive=False, elem_classes=["mono"])

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════
            # Tab 0: Examples Browser — load like Dataset Browser, I/O like H5
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Examples Browser"):
                gr.Markdown(
                    "### Examples Browser\n"
                    "Load the bundled **`examples/`** samples, pick one (gallery or slider), "
                    "edit the **task instruction** / **prompt style**, and run inference to get "
                    "the **Predicted 2D** trajectory and the **Full conversation**."
                )
                with gr.Row():
                    ex_dir = gr.Textbox(label="Examples Directory", value=EXAMPLES_DIR, scale=4)
                    ex_load_btn = gr.Button("Load Examples", variant="primary", scale=1)
                ex_status = gr.Textbox(label="Status", interactive=False, elem_classes=["mono"])

                ex_gallery = gr.Gallery(
                    label="Click a sample to select (thumbnails)",
                    columns=6, rows=1, height=170, object_fit="cover",
                )
                ex_idx = gr.Slider(
                    label="Sample # (use this to select; works without thumbnails)",
                    minimum=0, maximum=max(1, len(scan_examples(EXAMPLES_DIR)) - 1),
                    value=0, step=1,
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### Inputs")
                        ex_rgb = gr.Image(label="RGB Image", type="numpy", height=300)
                        ex_depth = gr.Image(label="Metric Depth", type="numpy", height=220)
                        ex_instruction = gr.Textbox(
                            label="Task Instruction (editable — defaults to the example's)",
                            interactive=True, lines=2,
                        )
                        ex_prompt_style = gr.Radio(
                            label="Prompt Style",
                            choices=V5_PROMPT_STYLES, value=V5_DEFAULT_STYLE,
                            info="Trajectory (3D/2D): manipulation waypoints | Pointing (2D/3D): object/region points | 2D Bounding Box: object box | General VQA: free-form answer",
                        )
                        ex_predict_btn = gr.Button("Run Inference", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### Outputs")
                        ex_pred_image = gr.Image(label="Predicted Trajectory (2D)", type="numpy", height=350)
                        ex_conversation = gr.Textbox(
                            label="Full Conversation", lines=12, interactive=False,
                            show_copy_button=True, elem_classes=["mono"],
                        )

                gr.Markdown("### 3D Scene + Trajectory (interactive)")
                ex_3d_plot = gr.Plot(label="3D Scene + Trajectory (rotate/zoom with mouse)")
                ex_current_idx = gr.State(value=0)

            # ══════════════════════════════════════════════════════════════
            # Tab 1: Manual Inference (original)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Manual Inference"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.Markdown("### Inputs")
                        image_input = gr.Image(label="RGB Image", type="pil", height=300)
                        depth_input = gr.File(label="Metric Depth (.npy, float32, meters)", file_types=[".npy"])
                        depth_preview = gr.Image(label="Depth Preview", type="numpy", height=200)
                        instruction_input = gr.Textbox(
                            label="Task Instruction",
                            placeholder="e.g. Pick up the red block and place it on the blue plate.",
                            lines=2,
                        )
                        manual_prompt_style = gr.Radio(
                            label="Prompt Style",
                            choices=V5_PROMPT_STYLES, value=V5_DEFAULT_STYLE,
                            info="Trajectory (3D/2D): manipulation waypoints | Pointing (2D/3D): object/region points | 2D Bounding Box: object box | General VQA: free-form answer",
                        )
                        predict_btn = gr.Button("Predict Trajectory", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### Outputs")
                        raw_output = gr.Textbox(label="Raw Model Output", lines=4, interactive=False,
                                                show_copy_button=True, elem_classes=["mono"])
                        manual_conversation = gr.Textbox(label="Full Conversation", lines=10, interactive=False,
                                                         show_copy_button=True, elem_classes=["mono"])
                        traj_image = gr.Image(label="2D Trajectory Overlay", type="numpy", height=350)
                        traj_3d = gr.Plot(label="3D Trajectory (interactive)")
                        with gr.Accordion("Parsed Waypoints", open=False):
                            wp_table = gr.Dataframe(
                                headers=["#", "u", "v", "depth (m)", "action"],
                                label="Waypoints", interactive=False,
                            )

                gr.Markdown("### Example Instructions")
                gr.Examples(
                    examples=[
                        ["Pick up the red block and place it on the blue plate."],
                        ["Slide the drawer open."],
                        ["Push the button on the left."],
                        ["Stack the green cube on top of the yellow cube."],
                        ["Grasp the mug by its handle and move it to the right."],
                    ],
                    inputs=[instruction_input],
                )

        # ── Events: Manual Inference ──

        def on_load(model_path):
            return server.load(model_path)

        def on_depth_upload(depth_file):
            if depth_file is None:
                return None
            try:
                depth = np.load(depth_file.name).astype(np.float32)
                return colorize_depth(depth)
            except Exception:
                return None

        def on_predict(image, depth_file, instruction, prompt_style):
            if image is None:
                return "Upload an RGB image.", "", None, None, None
            if depth_file is None:
                return "Upload a metric depth .npy file.", "", None, None, None
            if not instruction.strip():
                return "Enter a task instruction.", "", None, None, None
            if not server.is_loaded():
                return "Load the model first (expand Model Settings).", "", None, None, None

            tmp_dir = tempfile.mkdtemp(prefix="traj3d_")
            img_path, npz_path, rgb_resized, depth_resized = prepare_inputs(
                image, depth_file.name, tmp_dir
            )
            h, w = rgb_resized.shape[:2]

            human_msg = build_v5_human_message(instruction, prompt_style)
            raw = server.predict(img_path, npz_path, human_msg, system_prompt=V5_SYSTEM_PROMPT)
            conversation = format_conversation(V5_SYSTEM_PROMPT, f"<image>{human_msg}", raw)

            # General VQA → free-form answer, no trajectory to draw.
            if prompt_style == VQA_STYLE:
                return raw, conversation, rgb_resized, visualize_3d([], [], h, w), []

            # bbox (2D) → draw boxes instead of a trajectory.
            if prompt_style == BBOX_STYLE:
                boxes = parse_bbox_2d(raw)
                viz = visualize_bbox(rgb_resized, boxes)
                bbox_table = [[i, f"{b[0]:.0f},{b[1]:.0f}", f"{b[2]:.0f},{b[3]:.0f}", "", b[4]]
                              for i, b in enumerate(boxes)]
                return raw, conversation, viz, visualize_3d([], [], h, w), bbox_table

            # Pointing → independent numbered points (no connecting line).
            if prompt_style in POINTING_STYLES:
                pts, _ = parse_gt_structured_json(raw)
                viz = visualize_points(rgb_resized, pts) if pts else rgb_resized
                fig_3d = visualize_3d(pts, [None] * len(pts), h, w, connect=False)
                ptable = [[i + 1, f"{p[0]:.1f}", f"{p[1]:.1f}",
                           (f"{p[2]:.3f}" if len(p) > 2 and p[2] else ""), "point"]
                          for i, p in enumerate(pts)]
                return raw, conversation, viz, fig_3d, ptable

            waypoints, actions = parse_gt_structured_json(raw)
            if not waypoints:
                waypoints, actions = parse_trajectory(raw)
            viz_2d = visualize_2d(rgb_resized, waypoints, actions) if waypoints else rgb_resized
            fig_3d = visualize_3d(waypoints, actions, h, w)

            table = []
            for i, (wp, act) in enumerate(zip(waypoints, actions)):
                table.append([i, f"{wp[0]:.1f}", f"{wp[1]:.1f}", f"{wp[2]:.3f}", act or ""])

            return raw, conversation, viz_2d, fig_3d, table

        load_btn.click(fn=on_load, inputs=[model_path_box], outputs=[load_status])
        depth_input.change(fn=on_depth_upload, inputs=[depth_input], outputs=[depth_preview])
        predict_btn.click(
            fn=on_predict,
            inputs=[image_input, depth_input, instruction_input, manual_prompt_style],
            outputs=[raw_output, manual_conversation, traj_image, traj_3d, wp_table],
        )

        # ── Events: Examples Browser ──

        def _ex_display(sample):
            """Render an example: RGB, colorized depth, and the default instruction."""
            if sample is None:
                return None, None, ""
            rgb = cv2.cvtColor(cv2.imread(sample["rgb_path"]), cv2.COLOR_BGR2RGB)
            depth_color = None
            try:
                depth = np.load(sample["depth_npy_path"]).astype(np.float32)
                depth_color = colorize_depth(depth)
            except Exception:
                pass
            return rgb, depth_color, sample.get("instruction", "")

        def on_ex_load(examples_dir):
            status = examples_browser.load(examples_dir)
            if not examples_browser.samples:
                return status, [], None, None, "", 0, gr.update(maximum=1, value=0)
            gallery = [
                (s["rgb_path"], f"[{i}] {s.get('instruction', '')[:40]}")
                for i, s in enumerate(examples_browser.samples)
            ]
            rgb, depth_color, instr = _ex_display(examples_browser.get(0))
            n = len(examples_browser.samples)
            return (status, gallery, rgb, depth_color, instr, 0,
                    gr.update(maximum=max(1, n - 1), value=0))

        def on_ex_select_idx(idx):
            rgb, depth_color, instr = _ex_display(examples_browser.get(int(idx)))
            return rgb, depth_color, instr, int(examples_browser.current_idx)

        def on_ex_gallery_select(evt: gr.SelectData):
            rgb, depth_color, instr = _ex_display(examples_browser.get(evt.index))
            return rgb, depth_color, instr, int(examples_browser.current_idx)

        def on_ex_predict(current_idx, instruction, prompt_style):
            if not server.is_loaded():
                return None, "Load the model first (expand Model Settings).", None
            sample = (examples_browser.samples[int(current_idx)]
                      if examples_browser.samples else None)
            if sample is None:
                return None, "No example loaded — click 'Load Examples' first.", None
            if not instruction.strip():
                return None, "Enter a task instruction.", None
            try:
                rgb_pil = Image.open(sample["rgb_path"]).convert("RGB")
                tmp_dir = tempfile.mkdtemp(prefix="ex_traj_")
                img_path, npz_path, rgb_resized, _depth_resized = prepare_inputs(
                    rgb_pil, sample["depth_npy_path"], tmp_dir
                )
                h, w = rgb_resized.shape[:2]

                human_msg = build_v5_human_message(instruction, prompt_style)
                raw = server.predict(img_path, npz_path, human_msg, system_prompt=V5_SYSTEM_PROMPT)
                conversation = format_conversation(V5_SYSTEM_PROMPT, f"<image>{human_msg}", raw)

                # Real camera intrinsics from the example (falls back to an
                # assumed pinhole if the example ships no calibration), used to
                # build the scene point cloud that the trajectory is overlaid on.
                K = sample.get("intrinsics") or default_intrinsics(h, w)
                scene_pts, scene_cols = build_scene_pcd_simple(rgb_resized, _depth_resized, K)

                # General VQA → free-form answer; show the scene with no trajectory.
                if prompt_style == VQA_STYLE:
                    fig_3d = build_3d_scene_figure(scene_pts, scene_cols, {}, title=instruction.strip())
                    return rgb_resized, conversation, fig_3d
                # bbox (2D) → draw boxes instead of a trajectory.
                if prompt_style == BBOX_STYLE:
                    viz = visualize_bbox(rgb_resized, parse_bbox_2d(raw))
                    fig_3d = build_3d_scene_figure(scene_pts, scene_cols, {}, title=instruction.strip())
                    return viz, conversation, fig_3d

                # Pointing → independent numbered points (no connecting line). 2D points
                # get their depth sampled from the depth map so they place in the scene.
                if prompt_style in POINTING_STYLES:
                    pts, _ = parse_gt_structured_json(raw)
                    viz = visualize_points(rgb_resized, pts)
                    fig_3d = build_3d_scene_figure(scene_pts, scene_cols, {}, title=instruction.strip())
                    if pts:
                        arr = np.array(pts, dtype=np.float32)
                        if prompt_style == "2D Pointing" and _depth_resized is not None:
                            for i in range(len(arr)):
                                up = int(np.clip(round(arr[i, 0] / 1000 * w), 0, w - 1))
                                vp = int(np.clip(round(arr[i, 1] / 1000 * h), 0, h - 1))
                                arr[i, 2] = float(_depth_resized[vp, up])
                        xyz = _uvd_to_xyz(arr, K, w, h)
                        fig_3d.add_trace(go.Scatter3d(
                            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers+text",
                            marker=dict(size=8, color="lime", line=dict(width=2, color="white")),
                            text=[str(i + 1) for i in range(len(xyz))], textposition="top center",
                            name="Points", hoverinfo="skip"))
                    return viz, conversation, fig_3d

                pred_wp, pred_act = parse_gt_structured_json(raw)
                if not pred_wp:
                    pred_wp, pred_act = parse_trajectory(raw)
                pred_viz = visualize_2d(rgb_resized, pred_wp, pred_act) if pred_wp else rgb_resized

                traj_dict = {}
                if pred_wp:
                    pred_xyz = _uvd_to_xyz(np.array(pred_wp, dtype=np.float32), K, w, h)
                    traj_dict["Pred (red)"] = (pred_xyz, [1.0, 0.2, 0.0])
                fig_3d = build_3d_scene_figure(scene_pts, scene_cols, traj_dict, title=instruction.strip())
                return pred_viz, conversation, fig_3d
            except Exception:
                import traceback as _tb
                return None, "ERROR in on_ex_predict:\n" + _tb.format_exc(), None

        ex_load_btn.click(
            fn=on_ex_load,
            inputs=[ex_dir],
            outputs=[ex_status, ex_gallery, ex_rgb, ex_depth, ex_instruction, ex_current_idx, ex_idx],
        )
        ex_idx.change(
            fn=on_ex_select_idx,
            inputs=[ex_idx],
            outputs=[ex_rgb, ex_depth, ex_instruction, ex_current_idx],
        )
        ex_gallery.select(
            fn=on_ex_gallery_select,
            outputs=[ex_rgb, ex_depth, ex_instruction, ex_current_idx],
        )
        ex_predict_btn.click(
            fn=on_ex_predict,
            inputs=[ex_current_idx, ex_instruction, ex_prompt_style],
            outputs=[ex_pred_image, ex_conversation, ex_3d_plot],
        )

        # Auto-scan the examples folder on page load so the gallery is ready.
        demo.load(
            fn=on_ex_load,
            inputs=[ex_dir],
            outputs=[ex_status, ex_gallery, ex_rgb, ex_depth, ex_instruction, ex_current_idx, ex_idx],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="3D Trajectory Prediction Gradio App")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--no-share", action="store_true", help="Disable public share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--autoload", action="store_true", help="Load model on startup")
    args = parser.parse_args()

    demo = build_ui(args.model_path)

    if args.autoload:
        print(f"Auto-loading: model_path={args.model_path}")
        print(server.load(args.model_path))

    demo.queue().launch(
        server_name=args.host, server_port=args.port, share=not args.no_share,
        allowed_paths=[EXAMPLES_DIR],
    )


if __name__ == "__main__":
    main()
