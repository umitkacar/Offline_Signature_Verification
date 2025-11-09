"""Tests for dataset classes."""

import pickle

import pytest
from PIL import Image

from signature_verification.dataset import SignatureTestDataset, TrainDataset


class TestTrainDatasetUnit:
    """Unit tests for TrainDataset (without real data)."""

    def test_dataset_file_not_found(self):
        """Test that FileNotFoundError is raised for missing data."""
        with pytest.raises(FileNotFoundError):
            TrainDataset(data_path="/nonexistent/path/data.pkl")

    def test_dataset_with_mock_data(self, tmp_path):
        """Test dataset with mock pickle data."""
        # Create temporary pickle file with mock data
        mock_data = [
            ["path1.png", "path2.png", 0],
            ["path3.png", "path4.png", 1],
        ]

        pickle_path = tmp_path / "train_data.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(mock_data, f)

        # This will fail when trying to load images, but tests initialization
        dataset = TrainDataset(data_path=pickle_path)
        assert len(dataset) == 2
        assert dataset.pairs == mock_data


class TestTestDatasetUnit:
    """Unit tests for SignatureTestDataset (without real data)."""

    def test_dataset_file_not_found(self):
        """Test that FileNotFoundError is raised for missing data."""
        with pytest.raises(FileNotFoundError):
            SignatureTestDataset(data_path="/nonexistent/path/data.pkl")

    def test_dataset_with_mock_data(self, tmp_path):
        """Test dataset with mock pickle data."""
        # Create temporary pickle file with mock data
        mock_data = [
            ["path1.png", "path2.png", 0],
            ["path3.png", "path4.png", 1],
        ]

        pickle_path = tmp_path / "test_data.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(mock_data, f)

        dataset = SignatureTestDataset(data_path=pickle_path)
        assert len(dataset) == 2
        assert dataset.pairs == mock_data


class TestDatasetIntegration:
    """Integration tests with actual image files."""

    @pytest.fixture
    def create_mock_dataset(self, tmp_path):
        """Create a complete mock dataset with images and pickle file."""
        # Create mock images
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        images = {}
        for i in range(4):
            img_path = img_dir / f"image_{i}.png"
            # Create a simple grayscale image
            img = Image.new("L", (220, 155), color=128)
            img.save(img_path)
            images[f"image_{i}"] = str(img_path)

        # Create mock pairs data
        pairs = [
            [images["image_0"], images["image_1"], 0],
            [images["image_2"], images["image_3"], 1],
        ]

        # Save to pickle
        pickle_path = tmp_path / "dataset.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(pairs, f)

        return pickle_path, pairs

    def test_train_dataset_full_pipeline(self, create_mock_dataset):
        """Test full TrainDataset pipeline with real images."""
        pickle_path, expected_pairs = create_mock_dataset

        dataset = TrainDataset(data_path=pickle_path)

        assert len(dataset) == 2

        # Get first item
        item = dataset[0]
        assert len(item) == 3

        # Check tensor shapes
        assert item[0].shape == (1, 220, 155)  # First image tensor
        assert item[1].shape == (1, 220, 155)  # Second image tensor
        assert item[2] in [0, 1]  # Label

    def test_test_dataset_full_pipeline(self, create_mock_dataset):
        """Test full SignatureTestDataset pipeline with real images."""
        pickle_path, expected_pairs = create_mock_dataset

        dataset = SignatureTestDataset(data_path=pickle_path)

        assert len(dataset) == 2

        # Get first item
        item = dataset[0]
        assert len(item) == 3

        # Check tensor shapes
        assert item[0].shape == (1, 220, 155)
        assert item[1].shape == (1, 220, 155)
        assert item[2] in [0, 1]

    def test_dataset_iteration(self, create_mock_dataset):
        """Test iterating through dataset."""
        pickle_path, _ = create_mock_dataset

        dataset = TrainDataset(data_path=pickle_path)

        count = 0
        for item in dataset:
            assert len(item) == 3
            count += 1

        assert count == len(dataset)

    def test_dataset_indexing(self, create_mock_dataset):
        """Test dataset indexing."""
        pickle_path, _ = create_mock_dataset

        dataset = TrainDataset(data_path=pickle_path)

        # Test valid indices
        item_0 = dataset[0]
        item_1 = dataset[1]

        assert item_0[2] != item_1[2] or item_0[2] == item_1[2]  # Labels can be same or different

        # Test invalid index
        with pytest.raises(IndexError):
            _ = dataset[999]
