# Contributing to Offline Signature Verification

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/umitkacar/Offline_Signature_Verification.git
cd Offline_Signature_Verification
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements.txt
pip install -r requirements.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will run automatic checks before each commit:
- Code formatting (Black)
- Linting (Ruff)
- Type checking (mypy)
- Tests (pytest)

## Development Workflow

### Code Style

We use modern Python tooling:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **mypy** for type checking

Format your code before committing:

```bash
black src tests scripts
ruff check src tests scripts --fix
mypy src
```

### Type Hints

All new code should include type hints:

```python
def process_signature(image_path: Path) -> Tensor:
    """Process signature image.

    Args:
        image_path: Path to signature image

    Returns:
        Processed image tensor
    """
    ...
```

### Testing

#### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel
pytest -n auto

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestSiameseConvNet::test_forward_shape
```

#### Write Tests

All new features should include tests:

```python
def test_new_feature():
    """Test description."""
    # Arrange
    model = SiameseConvNet()

    # Act
    result = model.forward_once(input_tensor)

    # Assert
    assert result.shape == expected_shape
```

Test files should be placed in `tests/` with the naming pattern `test_*.py`.

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update README.md for major changes
- Add inline comments for complex logic

Example docstring:

```python
def calculate_distance(features_a: Tensor, features_b: Tensor) -> Tensor:
    """Calculate Euclidean distance between feature vectors.

    This function computes pairwise distances using PyTorch's efficient
    implementation optimized for GPU acceleration.

    Args:
        features_a: First feature tensor of shape (batch_size, feature_dim)
        features_b: Second feature tensor of shape (batch_size, feature_dim)

    Returns:
        Pairwise distances of shape (batch_size,)

    Raises:
        ValueError: If feature dimensions don't match

    Example:
        >>> feat_a = torch.randn(4, 128)
        >>> feat_b = torch.randn(4, 128)
        >>> dist = calculate_distance(feat_a, feat_b)
        >>> print(dist.shape)
        torch.Size([4])
    """
    ...
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write code following the style guide
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Review Guidelines

### For Contributors

- Keep PRs focused and atomic
- Write clear commit messages
- Respond to review comments promptly
- Update your PR based on feedback

### For Reviewers

- Be constructive and respectful
- Focus on code quality and maintainability
- Check for test coverage
- Verify documentation is updated

## Project Structure

```
Offline_Signature_Verification/
├── src/signature_verification/   # Main package
│   ├── __init__.py
│   ├── model.py                   # Neural network models
│   ├── dataset.py                 # Dataset classes
│   └── utils.py                   # Utility functions
├── tests/                         # Test suite
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_utils.py
├── scripts/                       # Standalone scripts
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── pyproject.toml                 # Project configuration
├── .pre-commit-config.yaml        # Pre-commit hooks
└── README.md                      # Main documentation
```

## Areas for Contribution

### High Priority

- [ ] Vision Transformer (ViT) implementation
- [ ] Swin Transformer integration
- [ ] Hybrid CNN-ViT architecture
- [ ] Few-shot learning capabilities
- [ ] Mobile deployment (ONNX/TensorRT)

### Medium Priority

- [ ] Data augmentation techniques
- [ ] Additional evaluation metrics
- [ ] Visualization tools
- [ ] Configuration file support
- [ ] Logging improvements

### Documentation

- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Performance benchmarks
- [ ] Deployment guides

## Questions or Issues?

- **Bug Reports**: [GitHub Issues](https://github.com/umitkacar/Offline_Signature_Verification/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/umitkacar/Offline_Signature_Verification/discussions)
- **Questions**: Open a discussion or issue

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Offline Signature Verification! 🎉
