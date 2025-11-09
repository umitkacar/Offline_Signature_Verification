"""Quick functionality test without data dependencies."""

import sys

import torch

print("=" * 60)
print("QUICK FUNCTIONALITY TEST")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from signature_verification import (
        ContrastiveLoss,
        SiameseConvNet,
        distance_metric,
        fix_pair_person,
        fix_pair_sign,
    )

    print("   ✅ All imports successful!")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test model initialization
print("\n2. Testing model initialization...")
try:
    model = SiameseConvNet()
    print(f"   ✅ Model initialized: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    sys.exit(1)

# Test loss function
print("\n3. Testing loss function...")
try:
    criterion = ContrastiveLoss(margin=2.0)
    print(f"   ✅ Loss function initialized with margin={criterion.margin}")
except Exception as e:
    print(f"   ❌ Loss initialization failed: {e}")
    sys.exit(1)

# Test forward pass
print("\n4. Testing forward pass...")
try:
    batch_size = 2
    x = torch.randn(batch_size, 1, 220, 155)
    y = torch.randn(batch_size, 1, 220, 155)

    model.eval()
    with torch.no_grad():
        f_x, f_y = model(x, y)

    assert f_x.shape == (batch_size, 128), f"Unexpected shape: {f_x.shape}"
    assert f_y.shape == (batch_size, 128), f"Unexpected shape: {f_y.shape}"
    print(f"   ✅ Forward pass successful! Output shape: {f_x.shape}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    sys.exit(1)

# Test loss calculation
print("\n5. Testing loss calculation...")
try:
    labels = torch.tensor([0, 1]).float()  # 0=same, 1=different
    loss = criterion(f_x, f_y, labels)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"   ✅ Loss calculation successful! Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Loss calculation failed: {e}")
    sys.exit(1)

# Test distance metric
print("\n6. Testing distance metric...")
try:
    distances = distance_metric(f_x, f_y)
    assert distances.shape == (batch_size,), f"Unexpected shape: {distances.shape}"
    assert (distances >= 0).all(), "Distances should be non-negative"
    print(f"   ✅ Distance metric successful! Distances: {distances.numpy()}")
except Exception as e:
    print(f"   ❌ Distance metric failed: {e}")
    sys.exit(1)

# Test utility functions
print("\n7. Testing utility functions...")
try:
    x1, y1 = fix_pair_person(5, 5)
    assert x1 != y1, "fix_pair_person should return different IDs"

    x2, y2 = fix_pair_sign(3, 3)
    assert x2 != y2, "fix_pair_sign should return different IDs"

    print("   ✅ Utility functions working correctly!")
except Exception as e:
    print(f"   ❌ Utility functions failed: {e}")
    sys.exit(1)

# Test CUDA availability
print("\n8. Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"   ✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
    try:
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        y_cuda = y.cuda()
        with torch.no_grad():
            f_x_cuda, f_y_cuda = model_cuda(x_cuda, y_cuda)
        print("   ✅ CUDA forward pass successful!")
    except Exception as e:
        print(f"   ⚠️  CUDA available but test failed: {e}")
else:
    print("   ℹ️  CUDA not available (CPU only)")

# Test gradient computation
print("\n9. Testing gradient computation...")
try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train = torch.randn(2, 1, 220, 155)
    y_train = torch.randn(2, 1, 220, 155)
    labels_train = torch.tensor([0, 1]).float()

    f_x_train, f_y_train = model(x_train, y_train)
    loss_train = criterion(f_x_train, f_y_train, labels_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    print(f"   ✅ Gradient computation successful! Loss: {loss_train.item():.4f}")
except Exception as e:
    print(f"   ❌ Gradient computation failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✅")
print("=" * 60)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Python version: {sys.version.split()[0]}")
print("\nThe package is working correctly and ready to use!")
print("=" * 60)
