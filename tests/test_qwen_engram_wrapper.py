"""QymyzForCausalLM grafting tests on a tiny stub base model (no pretrained weights)."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from kazllm.model.qwen_engram_wrapper import EngramConfig, EngramWrappedLayer, QymyzForCausalLM

HIDDEN = 32
VOCAB = 64
NUM_LAYERS = 4


class StubDecoderLayer(nn.Module):
    """Minimal HF-style decoder layer: returns a tuple like Qwen2DecoderLayer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        return (self.linear(hidden_states),)


class StubInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(StubDecoderLayer() for _ in range(NUM_LAYERS))


class StubBaseModel(nn.Module):
    """Mimics the parts of a HF causal LM that QymyzForCausalLM touches."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB, rms_norm_eps=1e-6)
        self.model = StubInnerModel()

    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        hidden = self.model.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return hidden


def tiny_engram_config(layer_indices: list[int]) -> EngramConfig:
    return EngramConfig(
        layer_indices=layer_indices,
        ngram_orders=[2, 3],
        num_heads=2,
        table_size=101,
        slot_dim=8,
        conv_kernel_size=4,
    )


def test_engram_config_defaults() -> None:
    cfg = EngramConfig()
    assert cfg.layer_indices == [2, 7]
    assert cfg.ngram_orders == [2, 3]


def test_grafting_wraps_requested_layers() -> None:
    model = QymyzForCausalLM(StubBaseModel(), tiny_engram_config([1, 3]))
    assert model._wrapped_indices == [1, 3]
    layers = model._get_decoder_layers()
    for idx in range(NUM_LAYERS):
        if idx in (1, 3):
            assert isinstance(layers[idx], EngramWrappedLayer)
        else:
            assert isinstance(layers[idx], StubDecoderLayer)


def test_out_of_range_layer_index_is_skipped() -> None:
    model = QymyzForCausalLM(StubBaseModel(), tiny_engram_config([2, 99]))
    assert model._wrapped_indices == [2]


def test_forward_shape_and_input_ids_cleanup() -> None:
    torch.manual_seed(0)
    model = QymyzForCausalLM(StubBaseModel(), tiny_engram_config([1]))
    input_ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.shape == (2, 8, HIDDEN)
    # Stashed input_ids must be cleared after forward (memory-leak guard)
    layers = model._get_decoder_layers()
    assert layers[1]._current_input_ids is None


def test_base_weights_are_preserved() -> None:
    base = StubBaseModel()
    before = {name: param.clone() for name, param in base.named_parameters()}
    QymyzForCausalLM(base, tiny_engram_config([0]))
    after = dict(base.named_parameters())
    for name, param in before.items():
        # In-place wrapping renames wrapped-layer params: layers.0.X -> layers.0.layer.X;
        # the weight VALUES must be untouched.
        new_name = (
            name if name in after else name.replace("model.layers.0.", "model.layers.0.layer.")
        )
        torch.testing.assert_close(after[new_name], param)


def test_engram_parameter_groups_partition() -> None:
    model = QymyzForCausalLM(StubBaseModel(), tiny_engram_config([1, 2]))
    all_params = {id(p) for p in model.engram_parameters()}
    table_params = {id(p) for p in model.engram_table_parameters()}
    other_params = {id(p) for p in model.engram_non_table_parameters()}
    assert all_params
    assert table_params
    assert table_params | other_params == all_params
    assert table_params & other_params == set()


def test_from_base_layer_indices_override() -> None:
    # Pass a tiny config: the default EngramConfig allocates ~500K-slot tables (too big for CI)
    model = QymyzForCausalLM.from_base(
        StubBaseModel(),
        engram_layer_indices=[0],
        engram_config=tiny_engram_config([1, 3]),
    )
    assert model._wrapped_indices == [0]
