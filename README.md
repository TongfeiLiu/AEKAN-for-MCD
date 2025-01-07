# AEKAN-for-MCD
The paper “AEKAN: Exploring Superpixel-based Autoencoder Kolmogorov-Arnold Networks for Unsupervised Multimodal Change Detection” ([Download: 10.1109/TGRS.2024.3515258](https://doi.org/10.1109/tgrs.2024.3515258)) has been published by IEEE Transactions on Geoscience and Remote Sensing. This repository contains the PyTorch implementation of AEKAN.

## Outline
<ul>
  <li>Introduction</li>
  <li>Requirements</li>
  <li>Installation</li>
  <li>Usage</li>
  <li>Parameters</li>
  <li>Results</li>
  <li>Example</li>
  <li>References</li>
  <li>Acknowledgements</li>
</ul>

## Introduction
Multimodal change detection involves identifying changes between images captured at different times and using different sensors (e.g., optical and SAR). **AEKAN** combines KAN to construct an autoencoder (AE), which can more effectively learn the commonality features of independent superpixel regions between modalities, thereby realizing multimodal change detection.
## Requirements
<ul>
  <li>Python 3.7 or higher</li>
  <li>PyTorch 11.7 or higher</li>
  <li>scipy</li>
  <li>NumPy</li>
  <li>skimage</li>
</ul>

## Installation
### 1. Clone the repository: git clone https://github.com/TongfeiLiu/AEKAN-for-MCD.git
cd AEKAN

### 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  

### 3. Install the required packages:
pip install -r requirements.txt

## Usage
### Prepare your data:
* First-time image (image_t1): e.g., SAR image.
* Second-time image (image_t2): e.g., optical image.
* Reference ground truth (Ref_gt): Ground truth change map for evaluation.
### Parameters settings:
* lr: Learning rate (default: 0.0001)
* weight_decay: Weight decay (default: 0.0001)
* N_SEG: The number of superpixels (varies depending on the data)
* Com: Compactness parameter for superpixel segmentation (varies depending on the data)
* epoch: Number of training epochs (default: 50)
### Results
After running the script, you will obtain:
* Change Intensity Maps (Visual representations of change intensities.)
* Binary Change Maps (Thresholded maps showing detected changes.)
* Performance Metrics (A val.txt file containing Overall Accuracy, Kappa coefficient, and F1 score, etc.)

## Example
Here’s a step-by-step example to get you started:

## References

## Acknowledgements
