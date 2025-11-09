"""Siamese Convolutional Neural Network for Signature Verification."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Dropout, Linear, LocalResponseNorm, MaxPool2d, Module


class SiameseConvNet(Module):
    """Siamese Convolutional Neural Network.

    This network uses shared weights to extract features from signature pairs
    and learns a similarity metric through contrastive loss.

    Architecture:
        - Conv1: 48 filters, 11x11 kernel
        - Conv2: 128 filters, 5x5 kernel
        - Conv3: 256 filters, 3x3 kernel
        - Conv4: 96 filters, 3x3 kernel
        - FC1: 1024 neurons
        - FC2: 128-dimensional embeddings
    """

    def __init__(self) -> None:
        """Initialize the Siamese Convolutional Network."""
        super().__init__()
        self.conv1 = Conv2d(1, 48, kernel_size=(11, 11), stride=1)
        self.lrn1 = LocalResponseNorm(48, alpha=1e-4, beta=0.75, k=2)
        self.pool1 = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv2 = Conv2d(48, 128, kernel_size=(5, 5), stride=1, padding=2)
        self.lrn2 = LocalResponseNorm(128, alpha=1e-4, beta=0.75, k=2)
        self.pool2 = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout1 = Dropout(0.3)
        self.conv3 = Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = Conv2d(256, 96, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout2 = Dropout(0.3)
        self.fc1 = Linear(25 * 17 * 96, 1024)
        self.dropout3 = Dropout(0.5)
        self.fc2 = Linear(1024, 128)

    def forward_once(self, x: Tensor) -> Tensor:
        """Forward pass through one branch of the Siamese network.

        Args:
            x: Input tensor of shape (batch_size, 1, 220, 155)

        Returns:
            Feature embedding of shape (batch_size, 128)
        """
        x = F.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.pool2(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout2(x)
        x = x.view(-1, 17 * 25 * 96)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through both branches of the Siamese network.

        Args:
            x: First signature tensor of shape (batch_size, 1, 220, 155)
            y: Second signature tensor of shape (batch_size, 1, 220, 155)

        Returns:
            Tuple of feature embeddings (f_x, f_y), each of shape (batch_size, 128)
        """
        f_x = self.forward_once(x)
        f_y = self.forward_once(y)
        return f_x, f_y


class ContrastiveLoss(Module):
    """Contrastive Loss for Siamese Networks.

    This loss function learns to minimize the distance between similar pairs
    and maximize the distance between dissimilar pairs.

    Args:
        margin: Margin for negative pairs (default: 2.0)
    """

    def __init__(self, margin: float = 2.0) -> None:
        """Initialize Contrastive Loss.

        Args:
            margin: Margin threshold for dissimilar pairs
        """
        super().__init__()
        self.margin = margin

    def forward(self, output1: Tensor, output2: Tensor, label: Tensor) -> Tensor:
        """Calculate contrastive loss.

        Args:
            output1: Features from first signature (batch_size, feature_dim)
            output2: Features from second signature (batch_size, feature_dim)
            label: Labels (0 for same person, 1 for different persons)

        Returns:
            Scalar loss value
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def distance_metric(features_a: Tensor, features_b: Tensor) -> Tensor:
    """Calculate Euclidean distance between feature vectors.

    Args:
        features_a: First feature tensor
        features_b: Second feature tensor

    Returns:
        Pairwise distances
    """
    return F.pairwise_distance(features_a, features_b)
