# AEKAN-for-MCD
The paper **TGRS2024** “AEKAN: Exploring Superpixel-based Autoencoder Kolmogorov-Arnold Networks for Unsupervised Multimodal Change Detection” ([Download: 10.1109/TGRS.2024.3515258](https://doi.org/10.1109/tgrs.2024.3515258)) has been published by **IEEE Transactions on Geoscience and Remote Sensing**. 

This repository is the PyTorch implementation of AEKAN.

## Outline
<ul>
  <li>Introduction</li>
  <li>Results Preview</li>
  <li>Requirements</li>
  <li>Installation</li>
  <li>Usage</li>
  <li>Citation</li>
  <li>Acknowledgements</li>
</ul>

## Introduction
Multimodal change detection involves identifying changes between images captured at different times and using different sensors (e.g., optical and SAR). **AEKAN** combines KAN to construct an autoencoder (AE), which can more effectively learn the commonality features of independent superpixel regions between modalities, thereby realizing changes between multimodal bi-temporal images. The framework of the proposed AEKAN is presented as follows:
![Framework of our proposed AEKAN)](https://github.com/TongfeiLiu/AEKAN-for-MCD/blob/main/Figs/Fig1-AEKAN.png)
### Characteristics of AEKAN
<ul>
  <li>Using superpixels as the unit of analysis</li>
  <li>Each superpixel is trained independently </li>
</ul>

## Results Preview  
Visual results of our proposed AEKAN and the other methods on the MCD dataset #1-#5: (a) IRG-McS, (b) GIR-MRF, (c) SCASC, (d) AGSCC, (e) IST-CRF, (f) GBF-CD, (g) GLSS, (h) CANet, (i) CACD, (j) SR-GCAE, (k) BAACL, and **(l) AEKAN (Ours)**. (Notation: green, red, white, and black color denote missed detection pixels, false detection pixels, correct detection changed pixels, correct detection unchanged pixels, respectively.
![Visual results of our proposed AEKAN and the other methods on the MCD dataset #1-#5: (a) IRG-McS, (b) GIR-MRF, (c) SCASC, (d) AGSCC, (e) IST-CRF, (f) GBF-CD, (g) GLSS, (h) CANet, (i) CACD, (j) SR-GCAE, (k) BAACL, and (l) AEKAN (Ours). (Notation: green color, red color, white color, and black color denote missed detection pixels, false detection pixels, correct detection changed pixels, correct detection unchanged pixels, respectively.)](https://github.com/TongfeiLiu/AEKAN-for-MCD/blob/main/Figs/Fig5-BCIs.png)

## Requirements
<ul>
  <li>python==3.10</li>
  <li>pytorch==2.4.0 </li>
  <li>scipy==1.14.0</li>
  <li>numpy==1.23.0</li>
  <li>scikit-image==0.24.0</li>
</ul>

## Installation
### 1. Clone the repository: git clone https://github.com/TongfeiLiu/AEKAN-for-MCD.git
cd AEKAN

### 2. Install the required packages:
pip install -r requirements.txt

## Usage
### 1. Prepare your data:
* First-time image (T1_img): e.g., SAR image.
* Second-time image (T2_img): e.g., optical image.
* Reference ground truth (GT_img): Ground truth change map for evaluation.

Please set up your folder like this:
```
rootdir\

data\dataset\

  |---- T1.png

  |---- T2.png

  |---- GT.png
```
### 2. Parameters Setup

* lr: Learning rate (default: 0.0001)
* weight_decay: Weight decay (default: 0.0001)
* N_SEG: The number of superpixels (default: 1400) (varies depending on the data)
* Com: Compactness parameter for superpixel segmentation (default: 20) (varies depending on the data)
* epoch: Number of training epochs (default: 50)

**Note**: The code script accepts several command line arguments to adjust the model and processing. In addition, our current version of the code is only applicable to single-channel and three-channel heterogeneous remote sensing images. If you need to process heterogeneous remote sensing images with other numbers of channels, please modify the network initialization part in the code script to ensure that the encoder's dimensionality reduction channels are consistent with the reconstruction channels of the decoder.
  
### 3. Run the script:
```
python main.py
```
This will give the results of training and testing.

## Citation
If you find our work useful for your research, please consider citing our paper:
```
@ARTICLE{AEKAN,
  author={Liu, Tongfei and Xu, Jianjian and Lei, Tao and Wang, Yingbo and Du, Xiaogang and Zhang, Weichuan and Lv, Zhiyong and Gong, Maoguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={AEKAN: Exploring Superpixel-Based AutoEncoder Kolmogorov-Arnold Network for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={63},
  number={},
  pages={1-14},
  keywords={Feature extraction;Sensors;Image sensors;Training;Remote sensing;Sensor phenomena and characterization;Land surface;Analytical models;Sun;Radar imaging;Commonality features;heterogeneous images;Kolmogorov-Arnold Network (KAN);multimodal change detection (MCD)},
  doi={10.1109/TGRS.2024.3515258}
}
```

## Acknowledgement
This code is borrowed from the depository[1,2]. We are very grateful for the contributions of all related codes [3,4,5]. In addition, we are also very grateful for the outstanding contributions of the publicly available MCD datasets [6,7,8].
```
[1] https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py.
[2] https://github.com/KindXiaoming/pykan.
[3] https://github.com/yulisun.
[4] https://github.com/ChenHongruixuan/SRGCAE
[5] https://github.com/llu025/Heterogeneous_CD
[6] https://sites.google.com/view/luppino/data.
[7] Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.
[8] Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/).
```

## Contact us 
If you have any problems when running the code, please do not hesitate to contact us. Thanks.  
E-mail: liutongfei_home@hotmail.com or liutongfei@sust.edu.cn

Date: Jan 7, 2025  
