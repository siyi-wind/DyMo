<div align="center">

<h1><a href="https://openreview.net/forum?id=PWhDUWRVhM&noteId=PWhDUWRVhM">Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification (ICLR 2026)</a></h1>

**[Siyi Du](https://scholar.google.com/citations?user=zsOt8MYAAAAJ&hl=en), [Xinzhe Luo](https://scholar.google.com/citations?user=l-oyIaAAAAAJ&hl=en&oi=ao), [Declan P. O'Regan](https://scholar.google.com/citations?user=85u-LbAAAAAJ&hl=en&oi=ao), and [Chen Qin](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=mTWrOqHOqjoC&pagesize=80&sortby=pubdate)** 

![](https://komarev.com/ghpvc/?username=siyi-windDyMo&label=visitors)
![GitHub stars](https://badgen.net/github/stars/siyi-wind/DyMo)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2503.06277-b31b1b.svg)](https://arxiv.org/abs/2601.22853)

</div>

![DyMo](./Images/overview.jpg)
<p align="center">(a-b) Evidence of the discarding-imputation dilemma: (a-1) vs. (a-2) recovery-free methods (e.g., ModDrop) learn less discriminative features because they ignore highly task-relevant missing modalities {M,T}; (b) recovery-based methods (e.g., MoPoE) generate unreliable reconstructions, e.g., low-fidelity (orange) or misaligned (yellow). (c) Our DyMo, which addresses the dilemma by dynamically fusing task-relevant recovered modalities, improving accuracy by 1.61% on PolyMNIST, 1.68% on MST, and 3.88% on CelebA (Tab 1).</p>

This repository provides the official PyTorch implementation of [Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](https://openreview.net/forum?id=PWhDUWRVhM&noteId=PWhDUWRVhM). 

In addition to our DyMo, we include implementations of multiple baseline and comparison models, such as M3Care, MUSE, MTL, MAP, PDF, QMF, and DynMM. Please refer to the paper for detailed descriptions of these models.

Contact: s.du23@imperial.ac.uk (Siyi Du)

If you find this repository helpful, please consider giving it a :star:.


## Updates
[**2026-02-08**] The arXiv paper and the code are released. 
[**2026-02-11**] Upload model weights to Hugging Face. 


## Contents
- [1 Requirements](#1-requirements)
- [2 Preparation](#2-preparation)
- [3 Training & Testing](#3-training--testing)
- [4 Checkpoints](#4-checkpoints)
- [5 Licence & Citation](#5-licence--citation)
- [6 Acknowledgements](#6-acknowledgements)

## 1 Requirements
This codebase is implemented with **Python 3.9.15**, **PyTorch 1.11.0**, **CUDA 11.3.1**, and **cuDNN 8**.

```sh
cd DyMo/
conda env create --file environment.yaml
conda activate dymo
pip install numpy==1.23.5
```

## 2 Preparation

### 2.1 Data Downloading
We conduct experiments on five multimodal datasets with diverse modality combinations:

| Dataset   | Classification Task                          | #Modality | Modality Type        | #Train   | #Val   | #Test  | #Class | 
|-----------|----------------------------------------------|-----------|----------------------|----------|--------|--------|--------|
| PolyMNIST | Digit                                        | 5         | RGB image            | 60,000   | 3,000  | 7,000  | 10     |
| MST       | Digit                                        | 3         | RGB image, Text      | 1,121,360| 60,000 | 140,000| 10     |
| CelebA    | Face attribute                               | 2         | RGB image, Text      | 162,770  | 19,962 | 19,867 | 2      |
| DVM       | Car model                                    | 2         | RGB image, Table     | 70,565   | 17,642 | 88,207 | 283    |
| UKBB      | Coronary artery disease (CAD)                | 2         | MR image, Table      | 3,482    | 6,510  | 3,617  | 2      |
| UKBB      | Myocardial infarction                        | 2         | MR image, Table      | 1,552    | 6,510  | 3,617  | 2      |

* The preparation of PolyMNIST, MNIST-SVHN-TEXT(MST), and CelebA follows [https://github.com/thomassutter/MoPoE](https://github.com/thomassutter/MoPoE). 

* Download the DVM dataset from [here](https://deepvisualmarketing.github.io/).

* Apply for UKBB access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) (Note that UKBB is semi-public and requires approval and access fees).

### 2.2 Data Pre-processing
For UKBB and DVM, we conduct the same preprocessing pipelines as in [siyi-wind/TIP](https://github.com/siyi-wind/TIP).

### 2.3 Modality Reconstructor Preparation
We evaluate DyMo with five modality recovery methods: MoPoE, MMVAE++, CMVAE, TIP, and Iterative Multivariate Imputer (IMI). 

| Dataset | Modality Recovery Model | Download Weights |
| --------|---------------------------------|---------------------------------|
| PolyMNIST | MoPoE, MMVAE++, CMVAE | [Folder](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/PolyMNIST) |
| MST       | MoPoE                          | [Folder](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/MST) |
| CelebA       | MoPoE                          | [Folder](hhttps://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/CelebA) |
| DVM          | TIP, IMI                            | [TIP](https://drive.google.com/file/d/1FPUfO-XNwlYb_YklIdi8vOHr5GjpcJvY/view?usp=sharing) |
| UKBB         | TIP, IMI                            | [TIP](https://drive.google.com/file/d/1AKUq64WXn3j6-IhoUwarRuZ2PVDgNg_g/view?usp=sharing) |

TIP weights are from our ECCV 2024 paper [siyiwind/TIP](https://github.com/siyi-wind/TIP), while other models are trained by ourselves. For IMI, we directly provide the imputed DVM tables in [datasets/IMI_imputed_data/DVM](./datasets/IMI_imputed_data/DVM/). Due to UKBB data policy, imputed UKBB tables are not released, but can be generated using [datasets/tabular_imputation_UKBB.ipynb](./datasets/tabular_imputation_UKBB.ipynb) when you have the access to the UKBB dataset. 

## 3 Training & Testing
We record the hyper-parameters used for each experiment under [configs/](./configs/) using the Hydra format, so it is very easy to reproduce models on different datasets. Below we provide some examples. 

### 3.1 DyMo

#### Step 1: Train DynamicTransformer
DynamicTransformer is the backbone of DyMo (Fig. 2 in the paper). Below is an example to train DynamicTransformer on CelebA.
```sh
cd DyMo/job_scripts
env CUDA_VISIBLE_DEVICES=0 bash CelebA_DynamicTransformer.sh
```

Trained model weights are provided in [4 Checkpoints](#4-checkpoints).

#### Step 2: Generate Gaussian Parameters for The ICS score 
After completing the training of DynamicTransformer, we need to generate and store class-wise means and variances for calculating the ICS score (Eq. 8 in the paper). Use [gaussian_parameter_generation/gaussian_COS.ipynb](./gaussian_parameter_generation/gaussian_COS.ipynb) and [gaussian_parameter_generation/gaussian_EU.ipynb](./gaussian_parameter_generation/gaussian_EU.ipynb) to generate gaussian parameters for euclidean distance and cosine distance, separately. 

Precomputed parameters are also available in [4 Checkpoints](#4-checkpoints).

#### Step 3: Run DyMo at the Test Dataset
DyMo is an inference-time method and is computationally efficient:
```sh
cd DyMo/job_scripts
env CUDA_VISIBLE_DEVICES=0 bash PolyMNIST_DyMo.sh
```

### Other Models
To train or evaluate other baseline models, modify the model name in the shell scripts under [job_scripts/](./job_scripts/). Available model configurations are listed in [configs/model/](./configs/model/).

## 4 Checkpoints
| Dataset | Download Model Checkpoints and Gaussian Parameters |
| --------|---------------------------------|
| PolyMNIST | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/PolyMNIST) |
| MST       | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/MST)                          |
| CelebA       | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/CelebA)                          |
| DVM          | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/DVM)                            |
| CAD        | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/CAD)                           |
| Infarction        | [Download](https://huggingface.co/siyiwind/DyMo/tree/main/model_checkpoints/Infarction)                           |



## 5 Licence & Citation

This repository is licensed under the Apache License, Version 2.

If you find this work useful, please cite:

```text
@inproceedings{du2026dymo,
  title={Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification},
  author={Du, Siyi and Luo, Xinzhe and O'Regan, Declan P. and Qin, Chen},
  booktitle={International Conference on Learning Representations (ICLR) 2026},
  year={2026}}
```


## 6 Acknowledgements
We would like to thank the following repositories for their valuable contributions:

* [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
* [MoPoE](https://github.com/thomassutter/MoPoE)
