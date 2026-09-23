import torch

from uwir.models import MobileNetUNet, build_model, parse_model_variant


def test_mobilenet_unet_matches_reported_ablation_parameter_count():
    model = MobileNetUNet(in_channels=5)

    assert sum(parameter.numel() for parameter in model.parameters()) == 3_704_055


def test_mobile_end_to_end_variants_are_registered_and_preserve_image_shape():
    image = torch.rand(1, 3, 32, 48)

    for name in (
        "learnable_latent_mobilenet_unet",
        "parameterized_physics_mobilenet_unet",
    ):
        spec = parse_model_variant(name)
        model = build_model(name).eval()

        assert (spec.in_channels, spec.physics_mode) == (3, "none")
        assert isinstance(model.enhancer, MobileNetUNet)
        with torch.no_grad():
            assert model(image).shape == image.shape
