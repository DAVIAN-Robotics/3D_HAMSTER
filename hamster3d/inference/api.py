"""Minimal Python API for 3D HAMSTER inference.

Usage:
    from hamster3d.inference.api import Hamster3DPredictor

    predictor = Hamster3DPredictor("path/to/ckpt")
    result = predictor.predict(rgb_pil, depth_npy, "Pick up the cup")
    print(result["waypoints"])
    print(result["raw_output"])
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .preprocessing import (
    TARGET_SIZE,
    build_geometry_inputs,
    build_v5_messages,
    prepare_inputs,
    resize_to_target,
)
from .postprocessing import parse_trajectory

logger = logging.getLogger(__name__)


class Hamster3DPredictor:
    """Inference predictor for 3D HAMSTER VLM.

    Loads a merged Qwen3-VL-Geometry model (no PEFT required) and provides
    a simple predict() interface for trajectory prediction.

    Args:
        model_path: Path to merged model directory
        device: CUDA device string (default: "cuda:0")
        dtype: Model dtype (default: torch.bfloat16)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the merged model and processor."""
        # Register custom model class
        from hamster3d.model import register_qwen3_vl_geometry
        try:
            register_qwen3_vl_geometry()
        except Exception:
            pass

        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading model from: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map=self.device,
        )
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {param_count:,} parameters on {self.device}")

    def predict(
        self,
        rgb: Image.Image,
        depth: np.ndarray,
        instruction: str,
        max_new_tokens: int = 1024,
        prompt_style: str = "v5",
    ) -> dict:
        """Run inference on a single RGB + depth input.

        Args:
            rgb: PIL RGB image (any resolution — auto-resized to 640 longest edge)
            depth: Metric depth map as float32 numpy array (H, W) in meters
            instruction: Task instruction (e.g., "Pick up the red cup")
            max_new_tokens: Maximum tokens to generate
            prompt_style: "v5" for structured JSON output, "v3" for <ans> tag output

        Returns:
            dict with keys:
                waypoints: list of [u, v, depth] coordinates
                actions: list of gripper actions (or None)
                raw_output: raw model output string
                rgb_resized: resized RGB numpy array
                depth_resized: resized depth numpy array
        """
        # Preprocess
        inputs = prepare_inputs(rgb, depth)
        rgb_resized = inputs["rgb_resized"]
        depth_resized = inputs["depth_resized"]

        # Build geometry inputs (RGB + depth tensors for the geometry encoder)
        geo_inputs = build_geometry_inputs(
            rgb_resized, depth_resized, device=self.device
        )

        # Build chat messages
        messages = build_v5_messages(instruction)

        # Process with Qwen3-VL processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Load the resized image for the processor
        rgb_pil_resized = Image.fromarray(rgb_resized)

        model_inputs = self.processor(
            text=[text],
            images=[rgb_pil_resized],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Match the training / Gradio precision: cast all floating model inputs
        # (e.g. pixel_values) to the model dtype. Feeding float32 into a bf16 model
        # changes vision-tower precision and drifts greedy decoding.
        mdtype = next(self.model.parameters()).dtype
        for _k, _v in list(model_inputs.items()):
            if torch.is_tensor(_v) and torch.is_floating_point(_v):
                model_inputs[_k] = _v.to(mdtype)

        # Add geometry inputs (also cast to the model dtype)
        model_inputs["geometry_encoder_inputs"] = [t.to(mdtype) for t in geo_inputs["geometry_encoder_inputs"]]
        model_inputs["depth_maps"] = [t.to(mdtype) for t in geo_inputs["depth_maps"]]

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode — skip prompt tokens
        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        raw_output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Parse trajectory
        waypoints, actions = parse_trajectory(raw_output, prompt_style)

        return {
            "waypoints": waypoints,
            "actions": actions,
            "raw_output": raw_output,
            "rgb_resized": rgb_resized,
            "depth_resized": depth_resized,
        }
