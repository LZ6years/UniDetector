from setuptools import setup, find_packages

setup(
    name="jittor-unidetector",
    version="0.1.0",
    description="Jittor implementation of UniDetector for universal object detection",
    author="LZ6years",
    author_email="lz6year3915@gmail.com",
    packages=find_packages(),
    install_requires=[
        "jittor>=1.3.8",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "clip-by-openai>=1.0",
        "opencv-python>=4.5.0",
        "pycocotools>=2.0.6",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "Pillow>=8.0.0",
        "albumentations>=1.1.0",
    ],
    python_requires=">=3.7",
) 