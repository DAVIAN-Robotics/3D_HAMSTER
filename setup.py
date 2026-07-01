from setuptools import setup, find_packages

setup(
    name="hamster3d",
    version="0.1.0",
    description="3D HAMSTER: Inference-only VLM for 3D trajectory prediction",
    # Includes the vendored LingBot-Depth encoder code under
    # hamster3d/lingbot_depth/mdm (Apache-2.0); see hamster3d/lingbot_depth/LICENSE.
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.57.0,<4.58.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.34.0,<1.0",
        "gradio>=4.0.0,<6.0.0",
        "plotly>=5.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "fire>=0.5.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "timm>=0.9.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "merge": ["peft>=0.15.0"],
        # Optional: matches the reference inference setup. The DINOv2 geometry
        # encoder uses xformers memory-efficient attention when available and
        # falls back to an equivalent vanilla path otherwise.
        "perf": ["xformers"],
    },
)
