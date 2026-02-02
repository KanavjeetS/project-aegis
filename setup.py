from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="project-aegis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Modal V-JEPA Architecture for Predictive Planetary Resilience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/project-aegis",
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "rl": [
            "gym>=0.26.0",
            "stable-baselines3>=2.1.0",
        ],
        "deploy": [
            "onnx>=1.15.0",
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aegis-extract=scripts.extract_embeddings:main",
            "aegis-train=scripts.train_vlm:main",
            "aegis-infer=scripts.inference_vlm:main",
        ],
    },
)
