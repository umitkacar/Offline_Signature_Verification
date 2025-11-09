"""Training script for Siamese Network signature verification."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from signature_verification.dataset import TrainDataset
from signature_verification.model import ContrastiveLoss, SiameseConvNet


def train(
    data_path: Path = Path("./Data/train_index.pkl"),
    model_dir: Path = Path("./Models"),
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    device: str = "auto",
) -> None:
    """Train the Siamese Network.

    Args:
        data_path: Path to training data pickle file
        model_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cuda', 'cpu', or 'auto')
    """
    # Setup device
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    device_obj = torch.device(device_name)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device_obj}")

    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    train_dataset = TrainDataset(data_path=data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(train_dataset)} pairs")

    # Initialize model, loss, and optimizer
    print("\nInitializing model...")
    model = SiameseConvNet().to(device_obj)
    criterion = ContrastiveLoss().to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        running_loss = 0.0

        for index, (images1, images2, labels) in enumerate(train_loader):
            # Move to device
            images1 = images1.to(device_obj)
            images2 = images2.to(device_obj)
            labels = labels.float().to(device_obj)

            # Forward pass
            features1, features2 = model(images1, images2)
            loss = criterion(features1, features2, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            total_loss += loss.item()
            running_loss += loss.item()

            # Print progress
            if (index + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{index+1}/{n_total_steps}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

        # Epoch summary
        epoch_avg_loss = total_loss / n_total_steps
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {epoch_avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = model_dir / f"checkpoint_epoch_{epoch}"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("\n✅ Training complete!")


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument(
        "--data", type=Path, default=Path("./Data/train_index.pkl"), help="Training data path"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("./Models"), help="Model save directory"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device"
    )

    args = parser.parse_args()

    train(
        data_path=args.data,
        model_dir=args.model_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
