import warnings

# Suppress harmless PyTorch JIT deprecation warning emitted by external packages (e.g. kornia)
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ["ALL_MODEL_NAMES", "ModelSpec", "build_model", "parse_model_variant"]


def __getattr__(name):
    if name in __all__:
        from . import models

        return getattr(models, name)
    raise AttributeError(name)
