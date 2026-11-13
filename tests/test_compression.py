import numpy as np
import pytest

from daf.forecasting.patch_pruning import PatchPruner

# -- patch pruning -----------------------------------------------------------


def test_split_pads_on_the_left():
    patches, pad = PatchPruner(patch_size=16).split(np.arange(100.0))
    assert patches.shape == (7, 16)
    assert pad == 12
    assert np.isnan(patches[0, :12]).all()
    assert patches[-1, -1] == 99.0


def test_budget_respects_ratio_and_minimum():
    p = PatchPruner(keep_ratio=0.25, min_recent=4)
    assert p.budget(32) == 8
    assert p.budget(8) == 4
    assert p.budget(2) == 2


@pytest.mark.parametrize("strategy", ["recent", "seasonal", "saliency"])
def test_pruned_context_is_whole_patches_in_order(strategy):
    rng = np.random.default_rng(0)
    x = rng.normal(size=512).cumsum().astype(np.float32)
    pruner = PatchPruner(keep_ratio=0.5, strategy=strategy, patch_size=16, season=48)

    keep = pruner.select(x)
    out = pruner(x)

    assert len(keep) == 16
    assert np.all(np.diff(keep) > 0)
    assert len(out) == 16 * 16
    # the newest patches always survive, so the context still ends at the origin
    np.testing.assert_array_equal(out[-64:], x[-64:])
    # every surviving patch is an unmodified slice of the original
    for j, i in enumerate(keep):
        np.testing.assert_array_equal(out[16 * j : 16 * (j + 1)], x[16 * i : 16 * (i + 1)])


def test_recent_equals_truncation():
    x = np.arange(160.0)
    out = PatchPruner(keep_ratio=0.5, strategy="recent", patch_size=16)(x)
    np.testing.assert_array_equal(out, x[-80:])


def test_keep_all_returns_context_unchanged():
    x = np.arange(50.0)
    np.testing.assert_array_equal(PatchPruner(keep_ratio=1.0)(x), x)


def test_padding_is_dropped_when_first_patch_survives():
    x = np.arange(40.0)  # 3 patches of 16, 8 steps of padding
    # patch 0 starts 48 steps (one season) before the origin, so it is kept
    pruner = PatchPruner(keep_ratio=0.6, strategy="seasonal", patch_size=16, min_recent=1,
                         season=48, horizon=16)
    keep = pruner.select(x)
    assert keep.tolist() == [0, 2]
    out = pruner(x)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out, np.r_[np.arange(8.0), np.arange(24.0, 40.0)])


def test_seasonal_prefers_patches_one_season_back():
    x = np.arange(16 * 12, dtype=float)
    pruner = PatchPruner(keep_ratio=0.25, strategy="seasonal", patch_size=16, min_recent=1,
                         season=64, horizon=16)
    # budget is 3 patches: the newest one, then those starting exactly one and
    # two seasons (4 and 8 patches) before the origin
    assert pruner.select(x).tolist() == [4, 8, 11]


def test_saliency_keeps_peaks_and_level_shifts():
    x = np.zeros(16 * 10)
    x[16 * 2 : 16 * 3] = 5 * np.sin(np.arange(16))
    pruner = PatchPruner(keep_ratio=0.3, strategy="saliency", patch_size=16, min_recent=1)
    assert 2 in pruner.select(x)

    shift = np.r_[np.zeros(80), np.full(80, 10.0)]
    assert {4, 5} <= set(pruner.select(shift).tolist())


def test_saliency_drops_empty_patches_first():
    x = np.r_[np.full(40, np.nan), np.arange(60.0)]
    keep = PatchPruner(keep_ratio=0.5, strategy="saliency", patch_size=10, min_recent=1).select(x)
    assert not set(keep.tolist()) & {0, 1, 2, 3}


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PatchPruner(keep_ratio=0.0)
    with pytest.raises(ValueError):
        PatchPruner(strategy="random")


# -- int8 quantization (needs torch) -----------------------------------------


def _tiny_model():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_patch_embedding = torch.nn.Linear(16, 64)
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)
            )
            self.output_patch_embedding = torch.nn.Linear(64, 9)

        def forward(self, x):
            return self.output_patch_embedding(self.encoder(self.input_patch_embedding(x)))

    torch.manual_seed(0)
    return torch, Tiny().eval()


def test_quantize_int8_skips_sensitive_layers():
    torch, model = _tiny_model()
    from daf.forecasting.quantization import count_linear_layers, model_size_mb, quantize_int8

    qmodel = quantize_int8(model)

    assert count_linear_layers(model) == {"float": 4, "int8": 0}
    assert count_linear_layers(qmodel) == {"float": 2, "int8": 2}
    assert model_size_mb(qmodel) < model_size_mb(model)

    x = torch.randn(8, 16)
    with torch.inference_mode():
        ref, out = model(x), qmodel(x)
    assert torch.allclose(ref, out, atol=0.05 * ref.abs().max().item())


def test_quantize_int8_everything():
    _, model = _tiny_model()
    from daf.forecasting.quantization import count_linear_layers, quantize_int8

    assert count_linear_layers(quantize_int8(model, skip=())) == {"float": 0, "int8": 4}
