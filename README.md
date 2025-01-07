# AEKAN-for-MCD
The paper “AEKAN: Exploring Superpixel-based Autoencoder Kolmogorov-Arnold Networks for Unsupervised Multimodal Change Detection” ([Download: 10.1109/TGRS.2024.3515258](https://doi.org/10.1109/tgrs.2024.3515258)) has been accepted by IEEE Transactions on Geoscience and Remote Sensing. The code and pre-trained parameters will be released soon.

#AEKAN: Exploring Superpixel-based AutoEncoder Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection
##
##This repository contains the PyTorch implementation of AEKAN: Exploring Superpixel-based AutoEncoder Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection (AEKAN).
#目录

<ul>

  <li>Introduction</li>


  <li>Requirements</li>


  <li>Installation</li>

  <li>Usage</li>

  <li>Parameters</li>

  <li>Results</li>

  <li>Example</li>

  <li>References</li>

</ul>

##Introduction
Multimodal change detection involves identifying changes between images captured at different times and using different sensors (e.g., optical and SAR).**AEKAN** combines KAN to construct an autoencoder (AE), which can more effectively learn the common features of independent superpixel regions between modalities, thereby realizing multimodal change detection.
##Requirements
<ul>

  <li>Python 3.7 or higher</li>

  <li>PyTorch 11.7 or higher</li>

  <li>scipy</li>

  <li>NumPy</li>

  <li>skimage</li>

</ul>
##Installation
###1. Clone the repository:
git clone https://github.com/yourusername/SDCGA.git

cd AEKAN

###2. Set up a virtual environment
python -m venv venv

source venv/bin/activate  
###3. Install the required packages:
pip install -r requirements.txt
##Usage
###1. Prepare your data:
* First-time image (image_t1): e.g., SAR image.
* Second-time image (image_t2): e.g., optical image.
* Reference ground truth (Ref_gt): Ground truth change map for evaluation.
##Parameters
* lr: Learning rate (default: 0.0001)
* weight_decay: Weight decay (default: 0.0001)
* N_SEG: The number of superpixels (varies depending on the data)
* Com: Compactness parameter for superpixel segmentation (varies depending on the data)
* epoch: Number of training epochs (default: 50)
##Result
After running the script, you will obtain:

Change Intensity Maps: Visual representations of change intensities.

Binary Change Maps: Thresholded maps showing detected changes.

Performance Metrics: A txt file containing Overall Accuracy, Kappa coefficient, and F1 score, etc.

Example
Here’s a step-by-step example to get you started:
