# Datasets Directory

This directory holds the training and evaluation datasets.
Due to file size limitations, dataset files are not tracked in git.

## Expected Directory Layout

```text
datasets/
├── EUVP/
│   ├── Paired/
│   │   ├── underwater_imagenet/
│   │   │   ├── trainA/      # Degraded underwater input images
│   │   │   └── trainB/      # Clean reference ground-truth images
│   │   ├── underwater_dark/
│   │   │   ├── trainA/
│   │   │   └── trainB/
│   │   └── underwater_scenes/
│   │       ├── trainA/
│   │       └── trainB/
│   └── test_samples/
│       ├── Inp/             # Test input images (515 images)
│       └── GTr/             # Test ground-truth images
│
└── UIEB/
    ├── raw-890/             # 890 real-world underwater degraded images
    └── reference-890/       # Corresponding reference images
```

## How to Prepare Datasets

1. **EUVP Dataset**:
   - Download the EUVP dataset from [EUVP Official Website / PapersWithCode](http://irvlab.cs.umn.edu/resources/euvp-dataset).
   - Extract the `Paired` folder and `test_samples` folder directly under `datasets/EUVP/`.

2. **UIEB Dataset**:
   - Download the UIEB dataset (890 raw and reference images) from [UIEB Official Source](https://li-chongyi.github.io/proj_benchmark.html).
   - Place the `raw-890` and `reference-890` folders directly under `datasets/UIEB/`.
