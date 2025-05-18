import torch

def test_torch_cuda_available():
    assert torch.cuda.is_available(), "CUDA is not available!"

def test_torch_version():
    version = torch.__version__
    assert version.startswith("2.0.1"), f"Expected Torch 2.0.1, got {version}"
