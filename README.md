# IPCV-Project
This is a repository for my implementation of a research paper based on image denoising
# SUNet: Swin Transformer UNet for Image Denoising  
Implementation and Evaluation (Simplified 3-Stage Version)

This repository contains the complete code used to implement, train, and evaluate a simplified version of **SUNet: Swin Transformer UNet for Image Denoising**, adapted from the original research paper by Fan *et al.* (2022).  

The implementation is part of the project submitted for **EC861 – Image Processing & Computer Vision**, National Institute of Technology Karnataka (NITK), Surathkal.

---

##  Overview

The original SUNet paper proposed combining the **Swin Transformer** (shifted window attention) with the **UNet** encoder–decoder architecture for high-performance image denoising on large-resolution datasets like DIV2K,CBSD68 (256*256 images).

Due to computational constraints, this repository implements a **lightweight 3-stage version** of SUNet designed for **64×64 images**, trained on the **STL10 dataset**.  
Despite its reduced depth, the model preserves the essential architectural ideas:
- Shallow feature extraction  
- Hierarchical Swin Transformer encoder  
- PixelShuffle-based decoder  
- Multi-scale representation learning  

The goal is to demonstrate SUNet’s effectiveness in a resource-constrained setting.

---

##  Features of This Implementation

- 3-stage Swin-UNet architecture (instead of 5-stage original)
- Simplified Swin Transformer blocks for small images
- Automatic upsampling factor calculation
- Evaluation with PSNR and MSE  
- Visualization utilities for noisy/clean/denoised images  
- Support for custom noise levels (default: σ = 25)

---

##  Performance (σ = 25)

| Metric | Value |
|--------|--------|
| **PSNR (dB)** | **28.10** |
| **MSE** | **1.640924e-03** |

---

## Citation
@inproceedings{fan2022sunet,
  title={SUNet: swin transformer UNet for image denoising},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={2333--2337},
  year={2022},
  organization={IEEE}
}
