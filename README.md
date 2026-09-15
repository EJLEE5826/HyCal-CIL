# HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning

[![arXiv](https://img.shields.io/badge/arXiv-2604.15678-b31b1b.svg)](https://arxiv.org/abs/2604.15678)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-1565C0.svg)](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_HyCal_A_Training-Free_Prototype_Calibration_Method_for_Cross-Discipline_Few-Shot_Class-Incremental_CVPR_2026_paper.html)

Official implementation of paper **HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning (CVPR 2026).**

HyCal is a training-free prototype calibration method for cross-discipline few-shot class-incremental learning. It uses frozen CLIP embeddings and combines cosine similarity with Mahalanobis distance to improve robustness under heterogeneous domains and imbalanced few-shot data.

## Installation

```bash
git clone https://github.com/EJLEE5826/HyCal-CIL.git HyCal-CIL
cd HyCal-CIL

conda create -n hycal python=3.9 -y
conda activate hycal
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the default package index does not match your system. The experiments in the paper used PyTorch 2.1.1 and a single NVIDIA GeForce RTX 3090 GPU.

## Dataset Setup

Run the following commands from the repository root. Store datasets under a shared directory; the reproduction scripts use `DATA_ROOT`, or you can pass it as `--data_root` to the runner.

```bash
export DATA_ROOT=/path/to/datasets
mkdir -p "$DATA_ROOT"
```

For the following datasets, follow the [CoOp dataset setup guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the images and split files under `DATA_ROOT`:

| Dataset | Config key |
| --- | --- |
| FGVC Aircraft | `aircraft` |
| Describable Textures (DTD) | `dtd` |
| EuroSAT | `eurosat` |
| Oxford Flowers 102 | `oxford_flowers` |
| Caltech101 | `caltech101` |
| Food101 | `food101` |
| Oxford-IIIT Pets | `oxford_pets` |
| Stanford Cars | `stanford_cars` |
| SUN397 | `sun397` |

### ArtBench

Use the [official 256x256 ImageFolder archive with train/test splits](https://github.com/liaopeiyuan/artbench#accessing-dataset). The loader expects `ArtBench10/artbench-10-imagefolder-split/{train,test}/` under `DATA_ROOT`.

```bash
mkdir -p "$DATA_ROOT/ArtBench10"
curl -fL https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar \
    -o "$DATA_ROOT/ArtBench10/artbench-10-imagefolder-split.tar"
tar -xf "$DATA_ROOT/ArtBench10/artbench-10-imagefolder-split.tar" -C "$DATA_ROOT/ArtBench10"
```

### MNIST

[Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) downloads automatically on the first run to `$DATA_ROOT/MNIST/raw/`.

### OrganAMNIST

Download and export the 224x224 version using [MedMNIST](https://medmnist.com/). This preserves the official splits and creates `organamnist_224/` and `organamnist_224.csv` under `MedMNIST/OrganAMNIST/`. Export only once into a new folder: the exporter appends CSV rows, so the command below stops if the output folder already exists.

```bash
python -m pip install medmnist
mkdir -p "$DATA_ROOT/MedMNIST/raw"
mkdir "$DATA_ROOT/MedMNIST/OrganAMNIST" && \
python -m medmnist save --flag=organamnist --size=224 --download=True \
    --root="$DATA_ROOT/MedMNIST/raw" --folder="$DATA_ROOT/MedMNIST/OrganAMNIST"
```

### Galaxy10 DECaLS

Download the HDF5 file from the [official dataset page](https://astronn.readthedocs.io/en/latest/galaxy10.html), then convert it to images and a split CSV under `Galaxy10_DECals/_galaxy_temp/`. The converter uses the existing stratified 80/10/10 split with seed 42 and refuses to overwrite an existing output folder.

```bash
mkdir -p "$DATA_ROOT/Galaxy10_DECals"
curl -fL https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5 \
    -o "$DATA_ROOT/Galaxy10_DECals/Galaxy10_DECals.h5"
python task_datasets/preprocess_galaxy10_decals.py --data-root "$DATA_ROOT"
```

The main XD-VSCIL domain order is Aircraft, ArtBench, DTD, EuroSAT, Galaxy10, MNIST, OrganAMNIST, and OxfordFlowers.

## Reproducing Paper Settings

The main paper uses frozen OpenAI CLIP ViT-B/16 features with image-text feature summation. Results are averaged over seeds `0`, `1`, `42`, and `1993`.

| Paper setting | Data config | Few-shot config | Script |
| --- | --- | --- | --- |
| High-Scale Domain Imbalance | `config/data_config.yaml` | `config/few_shot_config_highimbal.yaml` | `scripts/run_high_scale.sh` |
| Balanced-in-Class Domain | `config/data_config.yaml` | `config/few_shot_config_bal.yaml` | `scripts/run_balanced.sh` |
| Cross-Scale Imbalance | `config/data_config.yaml` | `config/few_shot_config_crossimbal_{seed}.yaml` | `scripts/run_cross_scale.sh` |
| Random domain order | `config/data_config_{0,1,42,1993}.yaml` | `config/few_shot_config_highimbal.yaml` | `scripts/run_random_order.sh` |
| X-TAIL | `config/data_config_xtail.yaml` | `config/few_shot_config_16shot.yaml` | `scripts/run_xtail.sh` |

Run commands from the repository root. Run a setting with:

```bash
bash scripts/run_high_scale.sh 0
```

The optional first argument is the GPU id. You can override paths with environment variables:

```bash
DATA_ROOT=/path/to/datasets OUTPUT_DIR=./results bash scripts/run_balanced.sh 0
```

## Single Run Example

```bash
python experiment_runner.py \
    --cfg config/data_config.yaml \
    --few_shot_cfg config/few_shot_config_highimbal.yaml \
    --data_root ./data \
    --gpu 0 \
    --model_provider openai \
    --clip_model ViT-B/16 \
    --fusion sum \
    --sim_metric weighted \
    --n_min_p 10 \
    --scale 5 \
    --seed 0 \
    --output_dir ./results
```

`--eval-setting` and `--eval_setting` are both accepted. W&B logging is disabled by default; enable it with `--use_wandb`.

Using pure `--sim_metric md` requires a precision matrix for every candidate class and fails if a class is unlearned or has no precision matrix.

## Outputs

Each run writes a timestamped directory under `--output_dir`:

```text
results/
  XD_v*/
    cil_8_sum_<timestamp>/
      models/
      plots/
      results.json
      task_accuracies.csv
      task_accuracies_edit.csv
      dataset_stats.json
      CDE_metrics.json
      experiment_log.log
```

`CDE_metrics.json` reports `S_CDE`, `S_adapt`, and `S_last`.

Transfer is `null` in `results.json` (`N/A` in logs) when unmeasured, because normal reproduction evaluates learned domains only.

## Project Structure

```text
HyCal-CIL/
  experiment_runner.py
  hycal/
    __init__.py
    experiment_setup.py
    experiment_workflow.py
    hycal_model.py
    hycal_evaluator.py
    metrics.py
    plotting.py
    dataloader_utils.py
  config/
  task_datasets/
  scripts/
  clip/
```

## Third-Party Code

This repository vendors a lightly modified copy of OpenAI CLIP in `clip/`. See `THIRD_PARTY_NOTICES.md` and `clip/LICENSE`.

## Citation

```bibtex
@article{lee2026hycal,
  title={HYCAL: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning},
  author={Lee, Eunju and Kim, MiHyeon and Kwon, JuneHyoung and Lee, Yoonji and Kim, JiHyun and Jang, Soojin and Kim, YoungBin},
  journal={arXiv preprint arXiv:2604.15678},
  year={2026}
}
```

## Acknowledgement

Our implementation builds on [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning) and benefits from [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp). We thank the authors for sharing their code.

## License

This project is released under the MIT License.
