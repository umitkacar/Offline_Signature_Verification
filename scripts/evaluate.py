"""Evaluation script with ROC and Precision-Recall curves."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from signature_verification.dataset import SignatureTestDataset
from signature_verification.model import SiameseConvNet, distance_metric


def calculate_metrics(
    predictions: NDArray, labels: NDArray, step: float = 0.001
) -> Tuple[List, List, List, List]:
    """Calculate ROC and Precision-Recall metrics across thresholds.

    Args:
        predictions: Distance predictions
        labels: Ground truth labels
        step: Threshold step size

    Returns:
        Tuple of (TPR_list, FPR_list, Precision_list, Recall_list)
    """
    threshold_max = np.max(predictions)
    threshold_min = np.min(predictions)
    P = np.sum(labels == 1)  # Positive samples (same person)
    N = np.sum(labels == 0)  # Negative samples (different persons)

    TPR_full = []
    FPR_full = []
    Precision_full = []
    Recall_full = []

    print(f"\nCalculating metrics from {threshold_min:.4f} to {threshold_max:.4f}...")
    print(f"Total positive samples: {P}, Total negative samples: {N}")

    for threshold in np.arange(threshold_min, threshold_max + step, step):
        # Classify: distance <= threshold means SAME person (positive)
        idx1 = predictions <= threshold
        idx2 = predictions > threshold

        TP = np.sum(labels[idx1] == 1)  # Correctly identified as same
        FN = P - TP  # Missed same persons
        TN = np.sum(labels[idx2] == 0)  # Correctly identified as different
        FP = N - TN  # Incorrectly identified as same

        # ROC metrics
        TPR = float(TP / P) if P > 0 else 0.0
        TNR = float(TN / N) if N > 0 else 0.0
        FPR = 1 - TNR

        TPR_full.append(TPR)
        FPR_full.append(FPR)

        # Precision-Recall metrics
        if TP > 0:
            Precision = float(TP / (TP + FP))
            Recall = float(TP / (TP + FN))
            Precision_full.append(Precision)
            Recall_full.append(Recall)

    return TPR_full, FPR_full, Precision_full, Recall_full


def evaluate(
    model_path: Path,
    data_path: Path = Path("./Data/test_index.pkl"),
    output_path: Path = Path("./evaluation_results.png"),
    batch_size: int = 8,
    device: str = "auto",
) -> None:
    """Evaluate model and generate ROC and PR curves.

    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to test data pickle file
        output_path: Path to save evaluation plot
        batch_size: Batch size for evaluation
        device: Device to use ('cuda', 'cpu', or 'auto')
    """
    # Setup device
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    device_obj = torch.device(device_name)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device_obj}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = SiameseConvNet().to(device_obj)
    model.load_state_dict(torch.load(model_path, map_location=device_obj))
    model.eval()

    # Load test dataset
    print(f"Loading test data from {data_path}...")
    test_dataset = SignatureTestDataset(data_path=data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)} pairs")

    # Calculate predictions
    print("\nCalculating predictions...")
    total_labels = []
    total_dist = []

    with torch.no_grad():
        for index, (images1, images2, labels) in enumerate(test_loader):
            images1 = images1.to(device_obj)
            images2 = images2.to(device_obj)

            features1, features2 = model(images1, images2)
            distances = distance_metric(features1, features2).cpu().numpy()

            total_dist.extend(distances)
            total_labels.extend(labels.numpy())

            if (index + 1) % 100 == 0:
                print(f"  Processed {(index + 1) * batch_size} pairs...")

    total_dist = np.array(total_dist)
    total_labels = np.array(total_labels).astype(int)

    print(f"\nTotal distances: {len(total_dist)}")
    print(f"Total labels: {len(total_labels)}")

    # Calculate metrics
    TPR, FPR, Precision, Recall = calculate_metrics(total_dist, total_labels)

    # Plot results
    print("\nGenerating plots...")
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(FPR, TPR, marker=".", label="ROC Curve", linewidth=2)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(Recall, Precision, marker=".", label="Precision-Recall", linewidth=2, color="orange")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved evaluation plot to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Distance range: [{np.min(total_dist):.4f}, {np.max(total_dist):.4f}]")
    print(f"Mean distance: {np.mean(total_dist):.4f}")
    print(f"Std distance: {np.std(total_dist):.4f}")
    print("=" * 50)

    print("\n✅ Evaluation complete!")


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Siamese Network")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data", type=Path, default=Path("./Data/test_index.pkl"), help="Test data path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./evaluation_results.png"),
        help="Output plot path",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device"
    )

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model checkpoint not found at {args.model}")
        exit(1)

    evaluate(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
