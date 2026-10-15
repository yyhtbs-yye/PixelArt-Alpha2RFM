import torch

from diffusion.utils.rfm_checkpoint import drop_to_empty_text
from diffusion.model.nets.PixArt_blocks import QKRMSNorm


def test_rmsnorm_uses_fp32_affine_and_has_no_bias():
    norm = QKRMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(.4, 1.8, 8))
    x = torch.randn(2, 3, 8).bfloat16()
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                * norm.weight.float()).bfloat16()
    torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)
    assert list(norm.state_dict()) == ['weight']


def test_empty_dropout_replaces_both_embeddings_and_mask():
    caption = torch.randn(3, 1, 120, 16)
    mask = torch.ones(3, 120, dtype=torch.bool)
    empty = torch.randn(1, 1, 120, 16)
    empty_mask = torch.zeros(1, 120, dtype=torch.bool)
    empty_mask[:, 0] = True
    for probability in (0., 1.):
        actual, actual_mask = drop_to_empty_text(caption, mask, empty, empty_mask, probability)
        torch.testing.assert_close(actual, empty.expand_as(caption) if probability else caption)
        assert torch.equal(actual_mask, empty_mask.expand_as(mask) if probability else mask)
