"""Tests for the Siamese CNN model."""

import pytest
import torch

from signature_verification.model import ContrastiveLoss, SiameseConvNet, distance_metric


class TestSiameseConvNet:
    """Test suite for SiameseConvNet."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = SiameseConvNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_once_shape(self):
        """Test forward_once output shape."""
        model = SiameseConvNet()
        model.eval()

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 1, 220, 155)

        with torch.no_grad():
            output = model.forward_once(x)

        assert output.shape == (batch_size, 128), f"Expected shape (4, 128), got {output.shape}"

    def test_forward_shape(self):
        """Test forward output shapes for both branches."""
        model = SiameseConvNet()
        model.eval()

        # Create dummy inputs
        batch_size = 4
        x = torch.randn(batch_size, 1, 220, 155)
        y = torch.randn(batch_size, 1, 220, 155)

        with torch.no_grad():
            f_x, f_y = model.forward(x, y)

        assert f_x.shape == (batch_size, 128)
        assert f_y.shape == (batch_size, 128)

    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        model = SiameseConvNet()
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test model works with different batch sizes."""
        model = SiameseConvNet()
        model.eval()

        x = torch.randn(batch_size, 1, 220, 155)
        y = torch.randn(batch_size, 1, 220, 155)

        with torch.no_grad():
            f_x, f_y = model.forward(x, y)

        assert f_x.shape == (batch_size, 128)
        assert f_y.shape == (batch_size, 128)


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss."""

    def test_loss_initialization(self):
        """Test loss function initializes with correct margin."""
        loss_fn = ContrastiveLoss(margin=2.0)
        assert loss_fn.margin == 2.0

    def test_loss_output_scalar(self):
        """Test that loss outputs a scalar value."""
        loss_fn = ContrastiveLoss()

        # Create dummy features
        batch_size = 4
        features1 = torch.randn(batch_size, 128)
        features2 = torch.randn(batch_size, 128)
        labels = torch.randint(0, 2, (batch_size,)).float()

        loss = loss_fn(features1, features2, labels)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_loss_same_pairs(self):
        """Test loss for identical pairs (should be near zero)."""
        loss_fn = ContrastiveLoss()

        batch_size = 4
        features = torch.randn(batch_size, 128)
        labels = torch.zeros(batch_size)  # Same person

        loss = loss_fn(features, features, labels)

        # Loss should be very small for identical features
        assert loss.item() < 0.1

    def test_loss_different_margin(self):
        """Test loss with different margin values."""
        features1 = torch.randn(4, 128)
        features2 = torch.randn(4, 128)
        labels = torch.ones(4)  # Different persons

        loss_m1 = ContrastiveLoss(margin=1.0)(features1, features2, labels)
        loss_m2 = ContrastiveLoss(margin=2.0)(features1, features2, labels)

        assert isinstance(loss_m1.item(), float)
        assert isinstance(loss_m2.item(), float)


class TestDistanceMetric:
    """Test suite for distance_metric function."""

    def test_distance_same_features(self):
        """Test distance for identical features (should be zero)."""
        features = torch.randn(4, 128)
        distance = distance_metric(features, features)

        # Distances should be very close to zero (within floating point precision)
        assert torch.allclose(distance, torch.zeros_like(distance), atol=1e-4)

    def test_distance_shape(self):
        """Test output shape of distance metric."""
        batch_size = 8
        features_a = torch.randn(batch_size, 128)
        features_b = torch.randn(batch_size, 128)

        distance = distance_metric(features_a, features_b)

        assert distance.shape == (batch_size,)

    def test_distance_non_negative(self):
        """Test that distances are non-negative."""
        features_a = torch.randn(4, 128)
        features_b = torch.randn(4, 128)

        distance = distance_metric(features_a, features_b)

        assert (distance >= 0).all()


class TestModelIntegration:
    """Integration tests for model components."""

    def test_full_training_step(self):
        """Test a complete training step."""
        model = SiameseConvNet()
        criterion = ContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy batch
        batch_size = 4
        images1 = torch.randn(batch_size, 1, 220, 155)
        images2 = torch.randn(batch_size, 1, 220, 155)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Forward pass
        features1, features2 = model(images1, images2)
        loss = criterion(features1, features2, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_model_cuda_compatibility(self):
        """Test model can be moved to CUDA if available."""
        model = SiameseConvNet()

        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(2, 1, 220, 155).cuda()
            y = torch.randn(2, 1, 220, 155).cuda()

            with torch.no_grad():
                f_x, f_y = model(x, y)

            assert f_x.is_cuda
            assert f_y.is_cuda
        else:
            # Just test CPU
            x = torch.randn(2, 1, 220, 155)
            y = torch.randn(2, 1, 220, 155)

            with torch.no_grad():
                f_x, f_y = model(x, y)

            assert not f_x.is_cuda
            assert not f_y.is_cuda
