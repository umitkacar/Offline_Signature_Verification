"""Offline Signature Verification - Deep Learning Package."""

from signature_verification.dataset import SignatureTestDataset, TrainDataset
from signature_verification.model import ContrastiveLoss, SiameseConvNet, distance_metric
from signature_verification.utils import (
    convert_to_image_tensor,
    fix_pair_person,
    fix_pair_sign,
    invert_image,
    invert_image_path,
)

__version__ = "2.0.0"
__all__ = [
    "SiameseConvNet",
    "ContrastiveLoss",
    "distance_metric",
    "TrainDataset",
    "SignatureTestDataset",
    "convert_to_image_tensor",
    "fix_pair_person",
    "fix_pair_sign",
    "invert_image",
    "invert_image_path",
]
