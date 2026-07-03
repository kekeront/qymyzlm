"""Offline tests for model-soup weight averaging — exact math on tiny tensors."""

import pytest
import torch
from qymyz_embed.merge import soup_state_dicts
from torch import nn


def test_exact_average_alpha_half() -> None:
    sd_a = {"w": torch.tensor([1.0, 3.0]), "b": torch.tensor([[2.0]])}
    sd_b = {"w": torch.tensor([2.0, 5.0]), "b": torch.tensor([[4.0]])}
    merged = soup_state_dicts(sd_a, sd_b)
    assert torch.equal(merged["w"], torch.tensor([1.5, 4.0]))
    assert torch.equal(merged["b"], torch.tensor([[3.0]]))


def test_exact_average_alpha_quarter() -> None:
    # powers of two -> exact float arithmetic: 0.25*4 + 0.75*8 = 7
    sd_a = {"w": torch.tensor([4.0])}
    sd_b = {"w": torch.tensor([8.0])}
    merged = soup_state_dicts(sd_a, sd_b, alpha=0.25)
    assert torch.equal(merged["w"], torch.tensor([7.0]))


def test_alpha_extremes() -> None:
    sd_a = {"w": torch.tensor([1.0])}
    sd_b = {"w": torch.tensor([9.0])}
    assert torch.equal(soup_state_dicts(sd_a, sd_b, alpha=1.0)["w"], sd_a["w"])
    assert torch.equal(soup_state_dicts(sd_a, sd_b, alpha=0.0)["w"], sd_b["w"])


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_out_of_range_raises(alpha: float) -> None:
    sd = {"w": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="alpha"):
        soup_state_dicts(sd, sd, alpha=alpha)


def test_key_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        soup_state_dicts({"w": torch.tensor([1.0])}, {"v": torch.tensor([1.0])})


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        soup_state_dicts({"w": torch.zeros(2)}, {"w": torch.zeros(3)})


def test_dtype_preserved_and_averaged_in_fp32() -> None:
    sd_a = {"w": torch.tensor([1.0], dtype=torch.float16)}
    sd_b = {"w": torch.tensor([2.0], dtype=torch.float16)}
    merged = soup_state_dicts(sd_a, sd_b)
    assert merged["w"].dtype == torch.float16
    assert torch.equal(merged["w"], torch.tensor([1.5], dtype=torch.float16))


def test_sources_not_mutated() -> None:
    sd_a = {"w": torch.tensor([1.0, 3.0])}
    sd_b = {"w": torch.tensor([2.0, 5.0])}
    soup_state_dicts(sd_a, sd_b)
    assert torch.equal(sd_a["w"], torch.tensor([1.0, 3.0]))
    assert torch.equal(sd_b["w"], torch.tensor([2.0, 5.0]))


def test_load_state_dict_roundtrip_on_modules() -> None:
    """The real-usage flow: state_dict() gives LIVE refs; soup + load must stay exact."""
    torch.manual_seed(0)
    module_a = nn.Linear(3, 2)
    module_b = nn.Linear(3, 2)
    expected = {
        k: 0.5 * v.detach().clone() + 0.5 * module_b.state_dict()[k].detach().clone()
        for k, v in module_a.state_dict().items()
    }
    merged = soup_state_dicts(module_a.state_dict(), module_b.state_dict())
    module_a.load_state_dict(merged)  # mutates module_a's live tensors
    for key, value in module_a.state_dict().items():
        assert torch.equal(value, expected[key]), key
    for key, value in merged.items():  # the merged dict itself must not have been corrupted
        assert torch.equal(value, expected[key]), key
