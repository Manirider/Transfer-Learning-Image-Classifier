import torch
import pytest
from src.model import TransferLearningClassifier

def test_transfer_learning_classifier_initialization():
    model = TransferLearningClassifier(num_classes=10, hidden_dim=256, dropout=0.3, pretrained=False)
    assert model.classifier[0].out_features == 256
    assert model.classifier[4].out_features == 10
    
def test_transfer_learning_classifier_forward():
    model = TransferLearningClassifier(num_classes=6, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)
    
def test_freeze_unfreeze_backbone():
    model = TransferLearningClassifier(pretrained=False)
    model.freeze_backbone()
    
    for param in model.backbone.parameters():
        assert not param.requires_grad
        
    model.unfreeze_backbone(from_layer=-5)
    
    params = list(model.backbone.parameters())
    for param in params[:-5]:
        assert not param.requires_grad
        
    for param in params[-5:]:
        assert param.requires_grad
