# A Simple Recipe for Language-guided Domain Generalized Segmentation
[Mohammad Fahes<sup>1</sup>](https://mfahes.github.io/),
[Tuan-Hung Vu<sup>1,2</sup>](https://tuanhungvu.github.io/),
[Andrei Bursuc<sup>1,2</sup>](https://abursuc.github.io/),
[Patrick Pérez<sup>1,2</sup>](https://ptrckprz.github.io/),
[Raoul de Charette<sup>1</sup>](https://team.inria.fr/rits/membres/raoul-de-charette/)</br>
<sup>1</sup> Inria, Paris, France

<sup>2</sup> valeo.ai, Paris, France <br>

Project page: https://astra-vision.github.io/FAMix/ <br />
Paper: https://arxiv.org/abs/2311.17922

TL; DR: FAMix (for Freeze, Augment, and Mix) is a simple method for domain generalized semantic segmentation, based on minimal fine-tuning, language-driven patch-wise style augmentation, and patch-wise style mixing of original and augmented styles.

## Citation
```
@InProceedings{fahes2024simple,
  title={A Simple Recipe for Language-guided Domain Generalized Segmentation},
  author={Fahes, Mohammad and Vu, Tuan-Hung and Bursuc, Andrei and P{\'e}rez, Patrick and de Charette, Raoul},
  booktitle={CVPR},
  year={2024}
}
```
# Demo
<p align="center">
  <b>Test on unseen youtube videos in different cities<br />
  Training dataset: GTA5 <br />
  Backbone: ResNet-50 <br />
  Segmenter: DeepLabv3+
  </b>
</p>
<p align="center">
  <img src="./demo/test_on_videos.gif" style="width:100%"/>
</p>

[**Watch the full video on YouTube**](https://www.youtube.com/embed/vyjtvx2El9Q?si=jr1BvOOMAAv3oAMG)

# Table of Content
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
- [Running FAMix](#running-famix)
  - [Style mining](#style-mining)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Inference & Visualization](#inference--visualization)
- [License](#license)
- [Acknowledgement](#acknowledgement)

# Installation
## Dependencies

First create a new conda environment with the required packages:
```
conda env create --file environment.yml
```

Then activate environment using:
```
conda activate famix_env
```

## Datasets

* **ACDC**: Download ACDC images and labels from [ACDC](https://acdc.vision.ee.ethz.ch/download). Please follow the dataset directory structure:
  ```html
  <ACDC_DIR>/                   % ACDC dataset root
  ├── rbg_anon/                 % input image (rgb_anon_trainvaltest.zip)
  └── gt/                       % semantic segmentation labels (gt_trainval.zip)
  ```
  
* **BDD100K**: Download BDD100K images and labels from [BDD100K](https://doc.bdd100k.com/download.html). Please follow the dataset directory structure:
  ```html
  <BDD100K_DIR>/              % BDD100K dataset root
  ├── images/                 % input image
  └── labels/                 % semantic segmentation labels
  ```

* **Cityscapes**: Follow the instructions in [Cityscapes](https://www.cityscapes-dataset.com/)
  to download the images and semantic segmentation labels. Please follow the dataset directory structure:
  ```html
  <CITYSCAPES_DIR>/             % Cityscapes dataset root
  ├── leftImg8bit/              % input image (leftImg8bit_trainvaltest.zip)
  └── gtFine/                   % semantic segmentation labels (gtFine_trainvaltest.zip)
  ```

* **GTA5**: Download GTA5 images and labels from [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/). Please follow the dataset directory structure:
  ```html
  <GTA5_DIR>/                   % GTA5 dataset root
  ├── images/                   % input image 
  └── labels/                   % semantic segmentation labels
  ```

* **Mapillary**: Download Mapillary images and labels from [Mapillary](https://www.mapillary.com/dataset/vistas). Please follow the dataset directory structure:
  ```html
  <MAPILLARY_DIR>/              % Mapillary dataset root
  ├── training                  % Training subset 
   └── images                     % input image
   └── labels                     % semantic segmentation labels
  ├── validation                % Validation subset
   └── images                     % input image
   └── labels                     % semantic segmentation labels

* **Synthia**: Download Synthia images and labels from [SYNTHIA-RAND-CITYSCAPES](https://synthia-dataset.net/downloads/) and split it following [SPLIT-DATA](https://github.com/shachoi/RobustNet/tree/main/split_data). Please follow the dataset directory structure:
  ```html
  <SYNTHIA>/                 % Synthia dataset root
  ├── RGB/                   % input image 
  └── GT/                    % semantic segmentation labels
  ```

# Running FAMix

## Style mining
```
python3 patch_PIN.py \
  --dataset <dataset_name> \
  --data_root <dataset_root> \
  --resize_feat \
  --save_dir <path_for_learnt_parameters_saving>
```

## Training
```
python3 main.py \
--dataset <dataset_name> \
--data_root <dataset_root> \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--ckpts_path <path_to_save_checkpoints> \
--path_for_stats <path_for_mined_styles>
```

## Evaluation
```
python3 main.py \
--dataset <dataset_name> \
--data_root <dataset_root> \
--ckpt <path_to_tested_model> \
--test_only \
--ACDC_sub <ACDC_subset_if_tested_on_ACDC>   
```

# Inference & Visualization
To test any model on any image and visualize the output, please add the images to predict_test directory and run:
``` 
python3 predict.py \
--ckpt <ckpt_path> \
--save_val_results_to <directory_for_saved_output_images>
```

# License
FAMix is released under the [Apache 2.0 license](./LICENSE).

# Acknowledgement
The code is based on this implementation of [DeepLabv3+](https://github.com/VainF/DeepLabV3Plus-Pytorch), and uses code from [CLIP](https://github.com/openai/CLIP), [PODA](https://github.com/astra-vision/PODA) and [RobustNet](https://github.com/shachoi/RobustNet).

---

[↑ back to top](#a-simple-recipe-for-language-guided-domain-generalized-segmentation)
