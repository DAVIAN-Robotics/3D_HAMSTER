"""LingBot-Depth geometry encoder for 3D HAMSTER.

Merged from base.py + lingbot_depth_encoder.py. Only LingBot-Depth is supported.
"""

import sys
import copy
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GeometryEncoderConfig:
    """Configuration for geometry encoders."""
    encoder_type: str = "lingbot_depth"
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    reference_frame: str = "first"
    feature_dim: int = 2048
    freeze_encoder: bool = True
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseGeometryEncoder(ABC, nn.Module):
    """Base class for geometry encoders."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        self.config = config
        self.reference_frame = config.reference_frame
        self.freeze_encoder = config.freeze_encoder

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        pass


def _maybe_add_local_lingbot_depth_to_syspath() -> bool:
    """Add lingbot-depth to sys.path if it exists locally.

    Searches in multiple locations:
    1. Next to this file's parent (3d_hamster/3d_hamster/model/../lingbot_depth/)
    2. In the repo root (3d_hamster/lingbot_depth/)
    """
    # Check relative to this file: ../lingbot_depth/
    model_dir = Path(__file__).resolve().parent
    candidate1 = model_dir.parent / "lingbot_depth"
    if candidate1.exists() and candidate1.is_dir():
        sys.path.insert(0, str(candidate1))
        return True

    # Check repo root
    repo_root = model_dir.parent.parent
    candidate2 = repo_root / "lingbot_depth"
    if candidate2.exists() and candidate2.is_dir():
        sys.path.insert(0, str(candidate2))
        return True

    return False


def _find_lingbot_checkpoint() -> Optional[str]:
    """Find the LingBot checkpoint in known locations."""
    model_dir = Path(__file__).resolve().parent

    candidates = [
        model_dir.parent / "lingbot_depth" / "checkpoints" / "model.pt",
        model_dir.parent.parent / "lingbot_depth" / "checkpoints" / "model.pt",
    ]

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_lingbot_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load LingBot model from checkpoint."""
    try:
        from mdm.model.v2 import MDMModel
    except ImportError:
        if _maybe_add_local_lingbot_depth_to_syspath():
            from mdm.model.v2 import MDMModel
        else:
            raise ImportError("LingBot-Depth not found. Cannot import `mdm`.")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_config = checkpoint['model_config']
    lingbot_model = MDMModel(**model_config)
    lingbot_model = lingbot_model.to_empty(device=device)
    lingbot_model.load_state_dict(checkpoint['model'], strict=False)

    # Verify a sample weight
    test_key = "encoder.backbone.blocks.0.attn.qkv.bias"
    if test_key in checkpoint['model']:
        ckpt_val = checkpoint['model'][test_key]
        model_val = lingbot_model.state_dict()[test_key].cpu()
        if model_val.abs().max().item() < 1e-6 and ckpt_val.abs().max().item() > 1e-6:
            raise RuntimeError(
                f"[LingBot] CRITICAL: Weight '{test_key}' is ALL ZEROS after loading!"
            )

    return lingbot_model


def build_lingbot_model_from_config(model_config: Dict[str, Any]) -> nn.Module:
    """Build a LingBot MDMModel *skeleton* from an embedded ``model_config``, WITHOUT
    loading any external checkpoint.

    The encoder structure (shapes/keys) is created here; the actual weights are
    expected to be filled afterward by the merged 3D HAMSTER checkpoint via
    ``from_pretrained`` (the bundled ``model.geometry_encoder.*`` tensors). This is
    what makes the released checkpoint self-contained — no separate ``model.pt`` and
    no network access are required.

    ``MDMModel.__init__`` builds the DINOv2 backbone with ``pretrained=False``, so no
    weights are downloaded at construction time.
    """
    try:
        from mdm.model.v2 import MDMModel
    except ImportError:
        if _maybe_add_local_lingbot_depth_to_syspath():
            from mdm.model.v2 import MDMModel
        else:
            raise ImportError("LingBot-Depth not found. Cannot import `mdm`.")

    return MDMModel(**model_config)


class LingBotDepthEncoder(BaseGeometryEncoder):
    """LingBot-Depth encoder wrapper.

    - Uses ViT-L/14 (patch size 14x14)
    - Processes RGB + Depth together
    - Encoder is ALWAYS frozen
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        # Embedded architecture config (self-contained path). When present, the
        # encoder is built from this dict and its weights come from the merged
        # checkpoint — no external model.pt is required.
        self.model_config = getattr(config, "model_config", None)

        if config.model_path:
            self.checkpoint_path = config.model_path
        elif self.model_config is not None:
            self.checkpoint_path = None
        else:
            found = _find_lingbot_checkpoint()
            if found:
                self.checkpoint_path = found
            else:
                raise FileNotFoundError(
                    "LingBot-Depth checkpoint not found. Set geometry_config.model_name_or_path "
                    "or provide geometry_config.encoder_model_config."
                )

        self.encoder: Optional[nn.Module] = None
        self._remap_depth_in: str = 'linear'
        self.patch_size = 14
        self._feature_dim: Optional[int] = None
        self._model_loaded = False

    def _get_target_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. LingBot-Depth requires GPU.")
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return torch.device(f'cuda:{local_rank}')
        except (ValueError, KeyError):
            return torch.device('cuda:0')

    def _ensure_model_loaded(self) -> None:
        if self._model_loaded:
            return

        if self.model_config is not None and not self.checkpoint_path:
            # Self-contained path: build structure from embedded config (no model.pt,
            # no network). Weights are filled later by from_pretrained.
            logger.info("[LingBot] Building encoder skeleton from embedded model_config "
                        "(weights from merged checkpoint)")
            lingbot_model = build_lingbot_model_from_config(self.model_config)
        else:
            target_device = self._get_target_device()
            logger.info(f"[LingBot] Loading encoder from: {self.checkpoint_path}")
            lingbot_model = load_lingbot_model(self.checkpoint_path, target_device)

        self.encoder = copy.deepcopy(lingbot_model.encoder)
        self._remap_depth_in = getattr(lingbot_model, 'remap_depth_in', 'linear')

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(f"[LingBot] Encoder extracted: {encoder_params:,} params (frozen)")

        del lingbot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_loaded = True

    def encode(
        self,
        images: torch.Tensor,
        depth_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("LingBot encoder not loaded. Call _ensure_model_loaded().")

        model_dtype = next(self.encoder.parameters()).dtype
        device = next(self.encoder.parameters()).device
        images = images.to(device=device, dtype=model_dtype)

        if depth_maps is None:
            depth_maps = torch.zeros(
                images.shape[0], images.shape[2], images.shape[3],
                device=device, dtype=model_dtype,
            )
        else:
            depth_maps = depth_maps.to(device=device, dtype=model_dtype)

        with torch.no_grad():
            features = self._extract_features(images, depth_maps)
        return features

    def _forward_feat(self, image, num_tokens, depth, **kwargs):
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth = depth.to(dtype=dtype, device=device)

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        features, cls_token, _, _ = self.encoder(
            image, depth, base_h, base_w,
            return_class_token=True,
            remap_depth_in=self._remap_depth_in,
            **kwargs
        )
        return features, cls_token

    @torch.inference_mode()
    def _infer_feat(self, image, depth_in=None, num_tokens=None, resolution_level=9, use_fp16=True, **kwargs):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        device = next(self.encoder.parameters()).device
        dtype = next(self.encoder.parameters()).dtype
        image = image.to(dtype=dtype, device=device)

        if depth_in is not None and depth_in.dim() == 2:
            depth_in = depth_in.unsqueeze(0).to(dtype=dtype, device=device)

        original_height, original_width = image.shape[-2:]

        if num_tokens is None:
            num_tokens = (original_height // self.patch_size) * (original_width // self.patch_size)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_fp16 and dtype != torch.bfloat16):
            features, cls_token = self._forward_feat(image, num_tokens=num_tokens, depth=depth_in, **kwargs)

        return features, cls_token

    def _extract_features(self, images, depth_maps):
        n_images, _, img_h, img_w = images.shape
        num_tokens = (img_h // self.patch_size) * (img_w // self.patch_size)

        features, cls_token = self._infer_feat(
            image=images, depth_in=depth_maps, num_tokens=num_tokens,
        )

        if isinstance(cls_token, torch.Tensor) and features.dim() == 4:
            features = features + cls_token[..., None, None]

        if features.dim() == 4:
            features = features.permute(0, 2, 3, 1).contiguous()
        elif features.dim() == 3:
            aspect_ratio = img_w / img_h
            base_h = round((num_tokens / aspect_ratio) ** 0.5)
            base_w = round((num_tokens * aspect_ratio) ** 0.5)
            features = features.view(n_images, base_h, base_w, features.shape[-1]).contiguous()

        return features

    def get_feature_dim(self) -> int:
        if self._feature_dim is None:
            self._ensure_model_loaded()
            if hasattr(self.encoder, 'embed_dim'):
                self._feature_dim = self.encoder.embed_dim
            elif hasattr(self.encoder, 'hidden_size'):
                self._feature_dim = self.encoder.hidden_size
            else:
                self._feature_dim = 1024
        return self._feature_dim

    def forward(self, images, depth_maps=None):
        return self.encode(images, depth_maps)

    def load_model(self, model_path=None):
        if model_path is not None:
            self.checkpoint_path = model_path
        self._model_loaded = False
        self.encoder = None
        self._feature_dim = None
        self._ensure_model_loaded()


def create_geometry_encoder(config: GeometryEncoderConfig) -> BaseGeometryEncoder:
    """Factory function — only LingBot-Depth is supported."""
    if config.encoder_type != "lingbot_depth":
        raise ValueError(f"Unsupported encoder type: {config.encoder_type}. Only 'lingbot_depth' is supported.")
    return LingBotDepthEncoder(config)
