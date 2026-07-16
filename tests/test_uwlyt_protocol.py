from pathlib import Path

from scripts.experiments import ablation_euvp, ablation_uieb

UWLYT_VARIANTS = {"uwlyt_3ch", "uwlyt_4ch_t", "uwlyt_4ch_b", "uwlyt_5ch"}


def test_euvp_runner_uses_matched_three_seed_legacy_protocol():
    args = ablation_euvp._make_parser().parse_args([])
    assert set(args.variants) >= UWLYT_VARIANTS
    assert args.num_runs == 3
    assert list(range(args.num_runs)) == [0, 1, 2]
    assert (args.nEpochs, args.batchSize, args.cropSize) == (50, 16, 256)
    assert (args.lr, args.weight_decay) == (1e-4, 1e-5)
    assert (args.scheduler_step, args.scheduler_gamma) == (30, 0.5)
    assert not args.cos_restart and not args.start_warmup
    assert (args.L1_weight, args.perceptual_weight, args.SSIM_weight) == (1.0, 1.0, 0.0)
    assert args.early_stop_patience == 20 and args.prior_method == "udcp"
    assert ablation_euvp.LEGACY_BASELINE == {
        "model": "unet_3ch",
        "psnr": 20.896,
        "ssim": 0.8857,
    }


def test_uieb_runner_retains_prior_selection_and_protocol():
    args = ablation_uieb._make_parser().parse_args([])
    assert set(args.variants) >= UWLYT_VARIANTS
    assert args.prior_methods == ["udcp", "gupdm"]
    assert args.num_runs == 3
    assert (args.nEpochs, args.batchSize, args.cropSize) == (50, 16, 256)
    assert (args.scheduler_step, args.scheduler_gamma) == (30, 0.5)
    assert (args.L1_weight, args.perceptual_weight, args.SSIM_weight) == (1.0, 1.0, 0.0)


def test_reproducible_shell_commands_use_legacy_spellings():
    root = Path(__file__).resolve().parents[1] / "scripts" / "experiments"
    required = (
        "--nEpochs",
        "--batchSize",
        "--cropSize",
        "--L1_weight",
        "--perceptual_weight",
        "--SSIM_weight",
    )
    for filename in ("run_ablation.sh", "run_ablation_uieb.sh"):
        command = (root / filename).read_text()
        assert all(flag in command for flag in required)
