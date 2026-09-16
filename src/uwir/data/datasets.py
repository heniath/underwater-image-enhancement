"""Paired UIEB and EUVP datasets used by the paper."""

import os
import random
from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms.functional as TF

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


from .utils import is_image_file, load_img

# ---------------------------------------------------------------------------
# UIEB  (890 real-world pairs; 800 train / 90 test split by convention)
# Expected layout:
#   <data_dir>/raw-890/    ← degraded inputs
#   <data_dir>/reference-890/  ← human-rated references (ground truth)
# ---------------------------------------------------------------------------


class UIEBDataset(data.Dataset):
    """
    Paired UIEB dataset for training and full-reference evaluation.

    Args:
        data_dir (str): Root directory that contains 'raw-890/' and
                        'reference-890/' sub-folders.
        transform: Torchvision transform applied to both images (e.g. Resize + ToTensor).
        augment (bool): Apply random aligned crop and hflip / vflip to both images
                        simultaneously (training only). Default: False.
    """

    INPUT_DIR = "raw-890"
    GT_DIR = "reference-890"

    def __init__(self, data_dir, transform=None, augment=False, in_memory=False, img_size=None):
        super().__init__()
        self.input_dir = join(data_dir, self.INPUT_DIR)
        self.gt_dir = join(data_dir, self.GT_DIR)
        self.transform = transform
        self.augment = augment
        self.in_memory = in_memory
        self.img_size = img_size

        # Stem-name matching (robust against ordering differences)
        gt_dict = {
            os.path.splitext(f)[0]: join(self.gt_dir, f)
            for f in listdir(self.gt_dir)
            if is_image_file(f)
        }
        self.input_files = []
        self.gt_files = []
        for f in sorted(listdir(self.input_dir)):
            if not is_image_file(f):
                continue
            stem = os.path.splitext(f)[0]
            if stem in gt_dict:
                self.input_files.append(join(self.input_dir, f))
                self.gt_files.append(gt_dict[stem])

        assert len(self.input_files) == len(self.gt_files), (
            f"UIEB: mismatched file counts "
            f"({len(self.input_files)} inputs vs {len(self.gt_files)} GTs)"
        )

        if self.in_memory:
            print("Loading UIEB dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="UIEB Inputs")]
            self.gt_images = [load_img(f) for f in tqdm(self.gt_files, desc="UIEB GTs")]

    def __getitem__(self, index):
        if self.in_memory:
            img_in = self.input_images[index]
            img_gt = self.gt_images[index]
        else:
            img_in = load_img(self.input_files[index])
            img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.img_size:
            img_in, img_gt = _paired_resize_crop(img_in, img_gt, self.img_size, self.augment)

        if self.augment:
            img_in, img_gt = _paired_augment(img_in, img_gt)

        if self.transform:
            img_in = self.transform(img_in)
            img_gt = self.transform(img_gt)

        return img_in, img_gt, file_in, file_gt

    def __len__(self):
        return len(self.input_files)

# ---------------------------------------------------------------------------
# Shared augmentation helper (coordinated on paired PIL images)
# ---------------------------------------------------------------------------


def _paired_resize_crop(img_in, img_gt, size: int, random_crop: bool):
    """Aspect-preserving resize followed by an aligned square crop."""
    if img_in.size != img_gt.size:
        raise ValueError(
            f"Paired images must have identical sizes, got {img_in.size} and {img_gt.size}"
        )
    img_in = TF.resize(img_in, size, antialias=True)
    img_gt = TF.resize(img_gt, size, antialias=True)
    width, height = img_in.size
    if random_crop:
        top = random.randint(0, max(0, height - size))
        left = random.randint(0, max(0, width - size))
    else:
        top = max(0, (height - size) // 2)
        left = max(0, (width - size) // 2)
    return (
        TF.crop(img_in, top, left, size, size),
        TF.crop(img_gt, top, left, size, size),
    )


def _paired_augment(img_in, img_gt):
    """
    Apply the same random geometric transforms to both images.
    Matches the notebook's _augment() method:
      - 50 % random horizontal flip
      - 50 % random vertical flip
    Rotation is intentionally omitted: its zero-filled borders corrupt the
    physics estimates and add artificial paired targets.

    Args:
        img_in, img_gt: PIL Images (after Resize, before ToTensor).
    Returns:
        Augmented (img_in, img_gt) PIL Images.
    """
    if random.random() > 0.5:
        img_in = TF.hflip(img_in)
        img_gt = TF.hflip(img_gt)
    if random.random() > 0.5:
        img_in = TF.vflip(img_in)
        img_gt = TF.vflip(img_gt)
    return img_in, img_gt


# ---------------------------------------------------------------------------
# EUVP  (~12 k paired images across several scene sub-sets)
# Expected layout (mirrors official EUVP release):
#   <data_dir>/Paired/underwater_imagenet/trainA/  ← degraded
#   <data_dir>/Paired/underwater_imagenet/trainB/  ← clean reference
# The sub-set name (e.g. 'underwater_imagenet', 'underwater_dark',
# 'underwater_scenes') is controlled via the `subset` argument.
# ---------------------------------------------------------------------------


class EUVPDataset(data.Dataset):
    """
    Large-scale paired EUVP dataset.  Used as the primary training corpus.

    Actual folder layout (Paired branch only has trainA / trainB):
        <data_dir>/Paired/<subset>/trainA/   ← degraded inputs
        <data_dir>/Paired/<subset>/trainB/   ← clean references
        <data_dir>/Paired/<subset>/validation/  ← unpaired (no GT)

    There is no testA / testB.  For a held-out validation set with ground
    truth, use torch.utils.data.random_split on the training data.

    Args:
        data_dir  (str): Root directory of the EUVP release
                         (the folder that contains 'Paired/').
        subset    (str | list[str]): One or more of
                         'underwater_imagenet' | 'underwater_dark' |
                         'underwater_scenes'.  Pass 'all' or a list to
                         combine multiple subsets (notebook default).
        transform: Applied to both input and GT images (e.g. Resize + ToTensor).
        augment (bool): Apply random aligned crop and hflip / vflip to both
                        images simultaneously (training only). Default: False.
    """

    SUBSETS = ("underwater_imagenet", "underwater_dark", "underwater_scenes")

    def __init__(
        self, data_dir, subset="all", transform=None, augment=False, in_memory=False, img_size=None
    ):
        super().__init__()

        # Resolve subset list
        if subset == "all":
            subsets = list(self.SUBSETS)
        elif isinstance(subset, str):
            # Support comma-separated: "underwater_dark,underwater_scenes"
            subsets = [s.strip() for s in subset.split(",")]
        else:
            subsets = list(subset)  # already a list

        for s in subsets:
            assert s in self.SUBSETS, f"subset must be one of {self.SUBSETS}, got '{s}'"

        self.transform = transform
        self.augment = augment
        self.in_memory = in_memory
        self.img_size = img_size
        self.input_files = []
        self.gt_files = []

        for s in subsets:
            input_dir = join(data_dir, "Paired", s, "trainA")
            gt_dir = join(data_dir, "Paired", s, "trainB")

            if not (os.path.isdir(input_dir) and os.path.isdir(gt_dir)):
                print(f"  [WARN] Missing: {input_dir} or {gt_dir} — skipping subset '{s}'")
                continue

            # Stem-name matching (robust against filename order differences)
            gt_dict = {
                os.path.splitext(f)[0]: join(gt_dir, f) for f in listdir(gt_dir) if is_image_file(f)
            }
            for f in sorted(listdir(input_dir)):
                if not is_image_file(f):
                    continue
                stem = os.path.splitext(f)[0]
                if stem in gt_dict:
                    self.input_files.append(join(input_dir, f))
                    self.gt_files.append(gt_dict[stem])

        if self.in_memory:
            print("Loading EUVP dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="EUVP Inputs")]
            self.gt_images = [load_img(f) for f in tqdm(self.gt_files, desc="EUVP GTs")]

    def __getitem__(self, index):
        if self.in_memory:
            img_in = self.input_images[index]
            img_gt = self.gt_images[index]
        else:
            img_in = load_img(self.input_files[index])
            img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.img_size:
            img_in, img_gt = _paired_resize_crop(img_in, img_gt, self.img_size, self.augment)

        if self.augment:
            img_in, img_gt = _paired_augment(img_in, img_gt)

        if self.transform:
            img_in = self.transform(img_in)
            img_gt = self.transform(img_gt)

        return img_in, img_gt, file_in, file_gt

    def __len__(self):
        return len(self.input_files)
