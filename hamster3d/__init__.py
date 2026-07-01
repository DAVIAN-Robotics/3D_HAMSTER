"""3D HAMSTER: Inference-only VLM for 3D trajectory prediction.

Usage:
    from hamster3d.inference import Hamster3DPredictor

    predictor = Hamster3DPredictor("path/to/ckpt")
    result = predictor.predict(rgb_pil, depth_npy, "Pick up the cup")
"""

__version__ = "0.1.0"
