import torch
import pytest

from models.tokenizer import VQ_VAE

def test_vqvae_instantiation():
    model = VQ_VAE()
    assert isinstance(model, VQ_VAE)

def test_vqvae_forward_output_shapes():
    batch_size = 2
    channels = 3
    height = 32
    width = 32
    model = VQ_VAE(channels=channels)
    x = torch.randn(batch_size, channels, height, width)
    decoded, indices, cmt_loss = model(x)
    assert decoded.shape == (batch_size, channels, height, width)
    assert indices is not None
    assert cmt_loss is not None

def test_vqvae_forward_types():
    model = VQ_VAE()
    x = torch.randn(1, 3, 32, 32)
    decoded, indices, cmt_loss = model(x)
    assert isinstance(decoded, torch.Tensor)
    assert isinstance(cmt_loss, torch.Tensor)