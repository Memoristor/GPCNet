# GPCNet

Official implementation of **"GPCNet: Grid-Parallel Compact Network With Kernel-Adaptive Feature Fusion for Semantic Segmentation of High-Resolution UAV Imagery"**, published in **IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026**.

[Paper](https://doi.org/10.1109/TGRS.2026.3725031) · [Environment Setup](ENVIRONMENTS.md) · [Configurations](configs) · [Citation](#-citation)

## 📋 Abstract

High-resolution UAV imagery poses two challenges for semantic segmentation: the computational cost of global self-attention and inconsistent feature representations in scenes with rich textures and fine structures. GPCNet combines a compact pretrained encoder, efficient grid-parallel feature encoding, and kernel-adaptive decoding to address both challenges. Its compact MiT-B2 variant, **GPCNet-P**, achieves **72.32% mIoU on UAVid** and **81.31% mIoU on VDD**, while removing 40% of the encoder parameters before segmentation training.

## 🔧 Framework Overview

[![GPCNet architecture: foresight encoder sparsification, grid-parallel encoding with stage-end semantic exchange, and kernel-adaptive feature fusion](assets/overview.png)](assets/overview.pdf)

GPCNet consists of three components:

1. **Foresight Encoder Sparsification (FES)** estimates parameter importance using the neural tangent kernel (NTK) trace on real UAV samples. It aggregates importance into dependency groups and iteratively removes redundant encoder parameters before training.
2. **Grid-Parallel Encoding with Stage-End Semantic Exchange (GPE-SSE)** processes nonoverlapping image grids through a shared encoder in parallel. The last Transformer block of each stage exchanges information across the full feature map to preserve cross-grid semantic context.
3. **Kernel-Adaptive Feature Fusion (KAFF)** progressively fuses multiscale features using spatially variable low-pass, high-pass, and dynamic dilated kernels, improving feature consistency and preserving fine structures.

The paper evaluates four variants:

| Variant   | Encoder | Encoder sparsity | Workflow               |
| --------- | ------- | ---------------- | ---------------------- |
| GPCNet    | MiT-B2  | 0%               | Train → Test          |
| GPCNet-P  | MiT-B2  | 40%              | Prune → Train → Test |
| GPCNet-S  | MiT-B1  | 0%               | Train → Test          |
| GPCNet-SP | MiT-B1  | 40%              | Prune → Train → Test |

The dense and compact variants share the same model class; the compact encoder is constructed by the pruning phase.

## 💻 Usage

### Installation

```bash
git clone https://github.com/Memoristor/GPCNet.git
cd GPCNet
```

Follow [ENVIRONMENTS.md](ENVIRONMENTS.md) to prepare the Python, PyTorch, CUDA, and NVIDIA DALI environment. Then build the local spatial variable convolution (SVC) extension in the same environment:

```bash
python -m pip install --no-build-isolation -e ./kernels/spatial_variable_convolution
python -c "from svc import spatial_variable_convolution; print('SVC extension imported successfully')"
```

The SVC operator requires an NVIDIA GPU and a CUDA toolkit with `nvcc` available for compilation. MiT encoders load ImageNet-pretrained weights through `segmentation_models_pytorch`; the weights must be downloaded or available in the local cache.

### Dataset Preparation

Download the datasets from their official project pages linked below, then set `dataset.root_path` in the selected YAML configuration to your dataset location. The supplied configurations use the following directories relative to that root:

| Dataset                                                        | Classes | Training images / labels                    | Validation images / labels              | Test images / labels                      |
| -------------------------------------------------------------- | ------- | ------------------------------------------- | --------------------------------------- | ----------------------------------------- |
| [UAVid](https://uavid.nl/)                                      | 8       | `uavid_train/seq*/Images` / `Labels`    | `uavid_val/seq*/Images` / `Labels`  | `test_gt/seq*/Images` / `Labels`      |
| [VDD](https://github.com/RussRobin/VDD)                         | 7       | `train/src` / `train/gt-dec`            | `val/src` / `val/gt-dec`            | `test/src` / `test/gt-dec`            |
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | 10      | `train/org-img` / `train/label-img-dec` | `val/org-img` / `val/label-img-dec` | `test/org-img` / `test/label-img-dec` |

<details>
<summary>Expected directory layouts</summary>

```text
UAVid/                         # dataset.root_path
├── uavid_train/
│   └── seq*/
│       ├── Images/*.png
│       └── Labels/*.png
├── uavid_val/
│   └── seq*/
│       ├── Images/*.png
│       └── Labels/*.png
└── test_gt/
    └── seq*/
        ├── Images/*.png
        └── Labels/*.png

VDD/                           # dataset.root_path
├── train/
├── val/
└── test/
    ├── src/*.JPG              # same layout in each split
    └── gt-dec/*.png

FloodNet/                      # dataset.root_path
├── train/
├── val/
└── test/
    ├── org-img/*.jpg          # same layout in each split
    └── label-img-dec/*.png
```

</details>

Labels must be **RGB color masks** matching the `PALETTE` in [datasets/uavid.py](datasets/uavid.py), [datasets/vdd.py](datasets/vdd.py), or [datasets/floodnet.py](datasets/floodnet.py). Convert class-index masks to these colors before using the DALI pipelines. Image and label files must correspond in sorted order. The UAVid `test_gt` directory is the layout expected by the local evaluation configuration and requires ground-truth labels; an images-only test release cannot be evaluated with this configuration.

Cropping is performed by the data pipeline. Training uses six overlapping `2048 × 2048` patches per image. Evaluation splits each image into two tiles along its width, resizes each tile to `2048 × 2048`, and stitches the resized-back predictions for evaluation at the original resolution.

### Configurations

Each configuration contains the `prune`, `train`, `valid`, and `test` phases:

| Dataset  | MiT-B1: GPCNet-S / GPCNet-SP                                                                                                                                                  | MiT-B2: GPCNet / GPCNet-P                                                                                                                                                     | Training epochs |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| UAVid    | [Config](configs/uavid/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_uavid.yaml)       | [Config](configs/uavid/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_uavid.yaml)       | 40              |
| VDD      | [Config](configs/vdd/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_vdd.yaml)           | [Config](configs/vdd/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_vdd.yaml)           | 40              |
| FloodNet | [Config](configs/floodnet/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep20_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_floodnet.yaml) | [Config](configs/floodnet/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep20_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_floodnet.yaml) | 20              |

Default settings shared by these configurations:

- **Encoder:** ImageNet-pretrained MiT, `4 × 4` grids, semantic exchange enabled at all four stages.
- **Decoder:** 64 channels, kernel size 3, 4 kernel groups, and two auxiliary segmentation heads.
- **FES:** 40% target encoder sparsity, 10 pruning iterations, and 8 sampled `1024 × 1024` patches.
- **Optimization:** AdamW with encoder learning rate `1e-4`, decoder learning rate `1e-3`, weight decay `0.05`, and cosine annealing.
- **Supervision:** cross-entropy with label smoothing `0.1` plus Dice loss; each auxiliary head has weight `0.4`.
- **Batch size:** 4 in total across training GPUs. Automatic mixed precision prefers BF16 when supported.

### Prune, Train, and Test

Run commands from the repository root. The wrapper accepts separate GPU lists and arguments for each phase:

```text
bash prune_train_test.sh CONFIG SEED PRUNE_GPUS PRUNE_ARGS TRAIN_GPUS TRAIN_ARGS TEST_GPUS TEST_ARGS
```

For example, run GPCNet-P on UAVid with one GPU for pruning, four GPUs for training, and one GPU for testing:

```bash
CONFIG="configs/uavid/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_uavid.yaml"
SEED=212

bash prune_train_test.sh "$CONFIG" "$SEED" \
    0 "-i -w GPCNet" \
    0,1,2,3 "-i -s -w GPCNet" \
    0 "-i -w GPCNet"
```

| Argument      | Meaning                                                                      |
| ------------- | ---------------------------------------------------------------------------- |
| `-i`        | Disable Weights & Biases logging. The examples do not require a W&B service. |
| `-s`        | Convert BatchNorm to SyncBatchNorm for training.                             |
| `-w GPCNet` | Set the project name, which also determines the output directory.            |
| `-r`        | Set the random seed when calling `main.py` directly.                       |

Pruning and standalone full-image evaluation currently require **one GPU**. Training supports multiple GPUs; the configured total batch size must be divisible by their number. For a single-GPU training run, replace `0,1,2,3` with `0`.

The pruning phase saves the compact model as `last_epoch.pth` at epoch 0. Training loads it automatically and updates the checkpoint; with the supplied configurations, the wrapper then evaluates the final `last_epoch.pth`. Each phase reads its checkpoint from the corresponding YAML `reload_pth` field. Keep the same configuration, seed, and project name across these phases. Existing checkpoints are resumed automatically.

### Evaluate a Checkpoint

Using the same `CONFIG` and `SEED` as above, set `test.reload_pth` to `best_iou.pth` in the existing YAML configuration to evaluate the checkpoint selected by validation IoU, then run:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    main.py -c "$CONFIG" -p test -r "$SEED" \
    -w GPCNet -i
```

Use `-p valid` for the validation split and select its checkpoint with `valid.reload_pth`. Set `test.reload_pth` to `last_epoch.pth` to evaluate the final checkpoint. A checkpoint filename is resolved within the experiment's `checkpoints/` directory; an absolute path can also be specified in YAML. The standalone evaluator reconstructs full-image predictions; the validation pass inside training computes metrics on the input tiles.

### Train a Dense Variant

Skip the pruning phase to train GPCNet or GPCNet-S. Use a separate project name to keep dense and pruned checkpoints in different directories:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    main.py -c "$CONFIG" -p train -r "$SEED" \
    -w GPCNet-Dense -s -i

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    main.py -c "$CONFIG" -p test -r "$SEED" \
    -w GPCNet-Dense -i
```

### Outputs

With the examples above, results are stored under:

```text
output/<project>/<dataset_class>/<model_class>/<config_stem>_seed_<seed>/
├── checkpoints/
│   ├── last_epoch.pth
│   ├── best_iou.pth
│   └── best_valid.json
├── logs/
├── wandb/
└── results/
    └── test/<checkpoint_stem>/
        ├── config.json
        ├── result.txt
        └── predictions/
```

`globals.output_root` controls the output root. The supplied configurations save complete model objects to preserve the pruned architecture; loading these checkpoints depends on the corresponding model definitions and module paths.

## 📈 Experimental Results

The following results are reported in **Tables II–V of the [paper](https://doi.org/10.1109/TGRS.2026.3725031)**. Accuracy is test-set mIoU (%). Efficiency is measured on a single NVIDIA RTX 3090 with BF16 precision, batch size 1, an input resolution of `2048 × 2048`, and the eight-class UAVid output head.

| Model     | Encoder | Encoder sparsity | UAVid           | VDD             | FloodNet        | Params (M)     | FLOPs (G)        | FPS             |
| --------- | ------- | ---------------- | --------------- | --------------- | --------------- | -------------- | ---------------- | --------------- |
| GPCNet    | MiT-B2  | 0%               | 72.28           | **81.35** | **78.07** | 25.11          | 948.76           | 12.36           |
| GPCNet-P  | MiT-B2  | 40%              | **72.32** | 81.31           | 77.61           | 15.39          | 665.13           | 14.12           |
| GPCNet-S  | MiT-B1  | 0%               | 70.95           | 80.44           | 76.73           | 14.07          | 762.36           | 15.60           |
| GPCNet-SP | MiT-B1  | 40%              | 70.65           | 79.94           | 76.21           | **8.76** | **493.69** | **18.88** |

For FloodNet, the paper averages metrics over the **nine foreground classes**, excluding Background. The configurations retain all ten output classes and export per-class metrics; average the nine foreground entries when comparing with the table. Metric values in `result.txt` are fractions rather than percentages. The values above are published results, not measurements from a new run of this checkout.

## 🎓 Citation

If you find this work useful, please cite our paper:

```bibtex

@article{deng_gpcnet_2026,
  author={Deng, Changjian and Cheng, Jian and Yang, Zhiguo and Zhang, Yijie and Xia, Ziying and Zhou, Zijing and Tashi, Nyima},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={GPCNet: Grid-Parallel Compact Network With Kernel-Adaptive Feature Fusion for Semantic Segmentation of High-Resolution UAV Imagery}, 
  year={2026},
  volume={64},
  number={},
  doi={10.1109/TGRS.2026.3725031}
}
```

## 📄 License

This repository includes the [GNU General Public License v3.0](LICENSE). Third-party components retain their respective license notices.
