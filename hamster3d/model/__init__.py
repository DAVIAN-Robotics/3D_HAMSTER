"""3D HAMSTER custom model: Qwen3-VL with LingBot-Depth geometry encoder."""

from .configuration import (
    Qwen3VLGeometryConfig,
    Qwen3VLGeometryTextConfig,
    Qwen3VLGeometryVisionConfig,
    GeometryEncoderConfig,
)
from .feature_fusion import (
    FeatureFusionConfig,
    FeatureFusionModule,
    GeometryFeatureMerger,
)
from .geometry_encoder import (
    BaseGeometryEncoder,
    GeometryEncoderConfig as _GEConfig,
    LingBotDepthEncoder,
    create_geometry_encoder,
)


def register_qwen3_vl_geometry():
    """Register the custom model with transformers AutoModel.

    After calling this, AutoModelForImageTextToText.from_pretrained() will
    correctly instantiate Qwen3VLGeometryForConditionalGeneration when the
    config has model_type="qwen3_vl_geometry".
    """
    from transformers import AutoConfig, AutoModelForImageTextToText
    from .modeling import Qwen3VLGeometryForConditionalGeneration

    AutoConfig.register("qwen3_vl_geometry", Qwen3VLGeometryConfig)
    AutoModelForImageTextToText.register(
        Qwen3VLGeometryConfig, Qwen3VLGeometryForConditionalGeneration
    )


# Auto-register on import (safe — fails silently if modeling.py not yet created)
try:
    register_qwen3_vl_geometry()
except Exception:
    pass

__all__ = [
    "Qwen3VLGeometryConfig",
    "Qwen3VLGeometryTextConfig",
    "Qwen3VLGeometryVisionConfig",
    "GeometryEncoderConfig",
    "FeatureFusionConfig",
    "FeatureFusionModule",
    "GeometryFeatureMerger",
    "BaseGeometryEncoder",
    "LingBotDepthEncoder",
    "create_geometry_encoder",
    "register_qwen3_vl_geometry",
]
