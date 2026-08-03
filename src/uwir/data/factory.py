"""Dataset factory for the UIEB and EUVP paper benchmarks."""

from torchvision.transforms import Compose, ToTensor

from .datasets import EUVPDataset, UIEBDataset


def _train_transform():
    return Compose([ToTensor()])


def get_euvp_training_set(
    data_dir: str,
    img_size: int = 256,
    subset: str = "all",
    in_memory: bool = False,
) -> EUVPDataset:
    return EUVPDataset(
        data_dir,
        subset=subset,
        transform=_train_transform(),
        augment=True,
        in_memory=in_memory,
        img_size=img_size,
    )


def get_uieb_training_set(
    data_dir: str,
    img_size: int = 256,
    in_memory: bool = False,
) -> UIEBDataset:
    return UIEBDataset(
        data_dir,
        transform=_train_transform(),
        augment=True,
        in_memory=in_memory,
        img_size=img_size,
    )
