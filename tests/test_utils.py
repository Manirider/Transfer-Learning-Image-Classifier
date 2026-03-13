import os
import torch
import pytest
from src.utils import set_seed, get_device, AverageMeter, EarlyStopping

def test_set_seed():
    set_seed(42)
    a = torch.rand(1)
    set_seed(42)
    b = torch.rand(1)
    assert torch.equal(a, b)

def test_get_device():
    device = get_device("cpu")
    assert device.type == "cpu"
    
def test_average_meter():
    meter = AverageMeter()
    assert meter.avg == 0.0
    meter.update(10.0, 2)
    meter.update(20.0, 2)
    assert meter.sum == 60.0
    assert meter.count == 4
    assert meter.avg == 15.0
    
def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
    es(0.1)
    assert not es.early_stop
    es(0.1) # No improvement
    assert not es.early_stop
    es(0.08) # Improvement
    assert not es.early_stop
    es(0.08) # No improvement
    assert not es.early_stop
    es(0.08) # No improvement, patience exceeded
    assert es.early_stop
