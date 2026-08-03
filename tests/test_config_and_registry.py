import pytest
import torch

from uwir.config import option
from uwir.models import ModelSpec, build_model, parse_model_variant


def test_model_spec_is_named_and_tuple_compatible():
    spec = parse_model_variant("unet_5ch")

    assert spec == ModelSpec("unet", 5, "tb")
    backbone, channels, physics_mode = spec
    assert (backbone, channels, physics_mode) == ("unet", 5, "tb")


def test_legacy_training_flags_are_the_only_config_namespace():
    parser = option()
    args = parser.parse_args(["--batchSize", "2", "--cropSize", "64", "--nEpochs", "3"])

    assert args.batchSize == 2
    assert args.cropSize == 64
    assert args.nEpochs == 3
    assert args.log_dir == "./logs/"
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "crop_size")
    assert not hasattr(args, "epochs")


@pytest.mark.parametrize("model_name", ["unet_5ch", "uwlyt_5ch", "uwlyttiny_5ch"])
def test_retained_models_preserve_image_shape(model_name):
    model = build_model(model_name, pretrained_backbone=False).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 5, 32, 32))
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
