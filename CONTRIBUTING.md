# Contributing to Project A.E.G.I.S.

Thank you for your interest in contributing! 🌍

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/yourusername/project-aegis/issues)
2. Create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment (OS, GPU, Python version)

### Suggesting Features

1. Open a [discussion](https://github.com/yourusername/project-aegis/discussions)
2. Describe the feature and use case
3. Wait for community feedback

### Code Contributions

#### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/project-aegis.git
cd project-aegis
git checkout - b feature/your-feature-name
```

#### 2. Set Up Environment

```bash
conda create -n aegis-dev python=3.9
conda activate aegis-dev
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev dependencies
```

#### 3. Make Changes

- Follow PEP 8 style guide
- Add docstrings to functions
- Write unit tests for new code
- Update documentation

#### 4. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Check code quality
black .
isort .
flake8 .
mypy models/ scripts/
```

#### 5. Submit Pull Request

1. Push to your fork
2. Create PR to `develop` branch (not `main`)
3. Fill in PR template
4. Wait for review

## Development Guidelines

### Code Style

- **Python:** PEP 8, max line length 100
- **Docstrings:** Google style
- **Type hints:** Required for public APIs

### Example

```python
def extract_embeddings(
    video_path: str,
    model: VJEPAModel,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract embeddings from a video file.
    
    Args:
        video_path: Path to video file
        model: Pre-trained V-JEPA model
        device: Device to run inference on
    
    Returns:
        embeddings: [T, embed_dim] tensor
    
    Raises:
        FileNotFoundError: If video_path doesn't exist
    """
    # Implementation...
```

### Testing

- **Unit tests** for individual functions
- **Integration tests** for end-to-end workflows
- **Minimum 80% code coverage** for new code

### Documentation

- Update README if adding features
- Add docstrings to new functions
- Create guides in `docs/` for major features

## Priority Areas for Contribution

1. **Additional Datasets**
   - Implement downloaders for disaster datasets
   - Create data augmentation strategies

2. **Model Optimizations**
   - Improve edge deployment (ONNX, TensorRT)
   - Quantization techniques

3. **RL Agent Training**
   - Implement Phase 4 (TD-MPC2)
   - Create Habitat-Sim environments

4. **Benchmarks**
   - Compare with BLIP-2, GPT-4V
   - Ablation studies

5. **Documentation**
   - More Colab notebooks
   - Video tutorials

## Code Review Process

1. Automated checks (CI/CD) must pass
2. At least one maintainer approval
3. No unresolved conversations
4. Up-to-date with `develop` branch

## Community

- [Discussions](https://github.com/yourusername/project-aegis/discussions)
- [Discord](https://discord.gg/aegis) (coming soon)
- [Twitter](https://twitter.com/project_aegis) (coming soon)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making A.E.G.I.S. better!** 🚀
