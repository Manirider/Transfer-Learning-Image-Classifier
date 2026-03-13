import os
import pytest
from unittest.mock import patch, MagicMock
from src.data_loader import get_dataloaders, prepare_splits

@pytest.fixture
def dummy_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create dummy splits
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        split_dir.mkdir()
        for cls in ["class1", "class2"]:
            cls_dir = split_dir / cls
            cls_dir.mkdir()
            
            # create 2 dummy images per class per split
            for i in range(2):
                img = cls_dir / f"img_{i}.png"
                img.touch()
                
    return {
        "project": {"seed": 42},
        "data": {
            "dataset_name": "dummy/dataset",
            "data_dir": str(data_dir),
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "image_size": 224,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False
        },
        "augmentation": {
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            "rotation_degrees": 15
        }
    }

@patch('src.augmentations.get_train_transforms', return_value=None)
@patch('src.augmentations.get_val_transforms', return_value=None)
@patch('src.data_loader.ImageFolder')
@patch('src.data_loader.DataLoader')
def test_get_dataloaders(mock_dataloader, mock_imagefolder, mock_val_trans, mock_train_trans, dummy_config):
    # Setup mock returns
    mock_imagefolder.return_value = MagicMock(spec=list)
    mock_imagefolder.return_value.__len__.return_value = 10
    
    train_loader, val_loader, test_loader = get_dataloaders(dummy_config)
    
    assert mock_imagefolder.call_count == 3
    assert mock_dataloader.call_count == 3

@patch('src.data_loader.kagglehub')
def test_prepare_splits(mock_kagglehub, tmp_path):
    source_dir = tmp_path / "source"
    seg_train = source_dir / "seg_train" / "seg_train"
    seg_train.mkdir(parents=True)
    
    for cls in ["class1", "class2"]:
        cls_dir = seg_train / cls
        cls_dir.mkdir()
        for i in range(10):  # 10 images per class
            (cls_dir / f"img_{i}.jpg").touch()
            
    dest_dir = tmp_path / "dest"
    
    config = {
        "project": {"seed": 42},
        "data": {
            "train_ratio": 0.7,
            "val_ratio": 0.15
        }
    }
    
    stats = prepare_splits(str(source_dir), str(dest_dir), config)
    
    assert "train" in stats
    assert "val" in stats
    assert "test" in stats
    
    # 70% of 10 is 7
    assert stats["train"]["class1"] == 7
    assert stats["train"]["class2"] == 7
    
    # 15% of 10 is 1
    assert stats["val"]["class1"] == 1
    assert stats["test"]["class1"] == 2 # Remaining
