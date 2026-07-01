# coding=utf-8
# Copyright 2025 The Qwen Team, HuggingFace Inc. team.
# Adapted for 3D HAMSTER inference-only repo.
#
# Licensed under the Apache License, Version 2.0

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Qwen3VLGeometryVisionConfig(PretrainedConfig):
    model_type = "qwen3_vl_geometry"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes if deepstack_visual_indexes is not None else [8, 16, 24]


class Qwen3VLGeometryTextConfig(PretrainedConfig):
    model_type = "qwen3_vl_geometry_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 5000000.0,
        rope_scaling: dict = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        rope_config_validation(self, ignore_keys={"mrope_section", "mrope_interleaved"})
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class GeometryEncoderConfig(PretrainedConfig):
    """Configuration for geometry encoder (LingBot-Depth)."""
    model_type = "geometry_encoder"
    base_config_key = "geometry_config"

    def __init__(
        self,
        enabled: bool = True,
        encoder_type: str = "lingbot_depth",
        model_name_or_path: str = None,
        encoder_model_config: dict = None,
        hidden_size: int = 1024,
        output_hidden_size: int = None,
        num_layers: int = 12,
        num_heads: int = 16,
        use_3d_position_encoding: bool = True,
        freeze_encoder: bool = False,
        fusion_method: str = "add",
        fusion_layers: list = None,
        merger_type: str = "mlp",
        merger_hidden_dim: int = None,
        reference_frame: str = "first",
        match_post_merge_resolution: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.encoder_type = encoder_type
        self.model_name_or_path = model_name_or_path
        # Embedded LingBot-Depth architecture config. When present, the encoder
        # structure is built directly from this dict (no external model.pt needed);
        # its weights are filled from the merged checkpoint's safetensors.
        self.encoder_model_config = encoder_model_config
        self.hidden_size = hidden_size
        self.output_hidden_size = output_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_3d_position_encoding = use_3d_position_encoding
        self.freeze_encoder = freeze_encoder
        self.fusion_method = fusion_method
        self.fusion_layers = fusion_layers
        self.merger_type = merger_type
        self.merger_hidden_dim = merger_hidden_dim
        self.reference_frame = reference_frame
        self.match_post_merge_resolution = match_post_merge_resolution


class Qwen3VLGeometryConfig(PretrainedConfig):
    """Configuration for Qwen3-VL with geometry encoder (inference-only)."""
    model_type = "qwen3_vl_geometry"
    sub_configs = {
        "vision_config": Qwen3VLGeometryVisionConfig,
        "text_config": Qwen3VLGeometryTextConfig,
        "geometry_config": GeometryEncoderConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict = None,
        vision_config: dict = None,
        geometry_config: dict = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        use_depth_decoder: bool = False,
        depth_loss_weight: float = 0.1,
        depth_only_training: bool = False,
        depth_drop_prob: float = 0.0,
        use_encoder_output_for_depth_loss: bool = False,
        save_depth_viz_dir: Optional[str] = None,
        save_depth_viz_interval: int = 100,
        save_depth_viz_max_per_dataset: int = 20,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(geometry_config, dict):
            self.geometry_config = self.sub_configs["geometry_config"](**geometry_config)
        elif geometry_config is None:
            self.geometry_config = self.sub_configs["geometry_config"]()
        else:
            self.geometry_config = geometry_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.use_depth_decoder = use_depth_decoder
        self.depth_loss_weight = depth_loss_weight
        self.depth_only_training = depth_only_training
        self.depth_drop_prob = depth_drop_prob
        self.use_encoder_output_for_depth_loss = use_encoder_output_for_depth_loss
        self.save_depth_viz_dir = save_depth_viz_dir
        self.save_depth_viz_interval = save_depth_viz_interval
        self.save_depth_viz_max_per_dataset = save_depth_viz_max_per_dataset
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = [
    "Qwen3VLGeometryConfig",
    "Qwen3VLGeometryTextConfig",
    "Qwen3VLGeometryVisionConfig",
    "GeometryEncoderConfig",
]
