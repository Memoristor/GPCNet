# Datasets and Configurations

[Back to README](../README.md) · [Environment Setup](ENVIRONMENTS.md)

This guide covers dataset preparation and the supplied experiment configurations. Run the commands in the README from the repository root.

## Dataset Preparation

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

Labels must be **RGB color masks** matching the `PALETTE` in [datasets/uavid.py](../datasets/uavid.py), [datasets/vdd.py](../datasets/vdd.py), or [datasets/floodnet.py](../datasets/floodnet.py). Convert class-index masks to these colors before using the DALI pipelines. Image and label files must correspond in sorted order. The local UAVid evaluation configuration expects a `test_gt` directory containing both images and ground-truth labels. It cannot evaluate a test release that contains only images.

Cropping is performed by the data pipeline. Training uses six overlapping `2048 × 2048` patches per image. Evaluation splits each image into two tiles along its width and resizes each tile to `2048 × 2048`. Predictions are resized back to the original tile dimensions and stitched together for evaluation at the original image resolution.

## Configurations

Each configuration contains the `prune`, `train`, `valid`, and `test` phases:

| Dataset  | MiT-B1: GPCNet-S / GPCNet-SP                                                                                                                                                  | MiT-B2: GPCNet / GPCNet-P                                                                                                                                                     | Training epochs |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| UAVid    | [Config](../configs/uavid/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_uavid.yaml)       | [Config](../configs/uavid/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_uavid.yaml)       | 40              |
| VDD      | [Config](../configs/vdd/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_vdd.yaml)           | [Config](../configs/vdd/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep40_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_vdd.yaml)           | 40              |
| FloodNet | [Config](../configs/floodnet/gpcnet_mit_b1/gpcnet_mit_b1_2048x2048_bs4_ep20_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_floodnet.yaml) | [Config](../configs/floodnet/gpcnet_mit_b2/gpcnet_mit_b2_2048x2048_bs4_ep20_adamw_lr1e-4_wd_5e-2_semseg_decoder_1e-3_wd_5e-2_cosine_lsce_1e-1_dice_highreso_semseg_floodnet.yaml) | 20              |

Default settings shared by these configurations:

- **Encoder:** ImageNet-pretrained MiT, `4 × 4` grids, semantic exchange enabled at all four stages.
- **Decoder:** 64 channels, kernel size 3, 4 kernel groups, and two auxiliary segmentation heads.
- **FES:** 40% target encoder sparsity, 10 pruning iterations, and 8 sampled `1024 × 1024` patches.
- **Optimization:** AdamW with encoder learning rate `1e-4`, decoder learning rate `1e-3`, weight decay `0.05`, and cosine annealing.
- **Supervision:** cross-entropy with label smoothing `0.1` and Dice loss. Each auxiliary head has weight `0.4`.
- **Batch size:** 4 in total across training GPUs. Automatic mixed precision prefers BF16 when supported.
