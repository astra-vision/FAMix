# A Simple Recipe for Language-guided Domain Generalized Segmentation
[Mohammad Fahes<sup>1</sup>](https://mfahes.github.io/),
[Tuan-Hung Vu<sup>1,2</sup>](https://tuanhungvu.github.io/),
[Andrei Bursuc<sup>1,2</sup>](https://abursuc.github.io/),
[Patrick Pérez<sup>3</sup>](https://ptrckprz.github.io/),
[Raoul de Charette<sup>1</sup>](https://team.inria.fr/rits/membres/raoul-de-charette/)</br>
<sup>1</sup> Inria, <sup>2</sup> valeo.ai, <sup>3</sup> Kyutai <br>

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



⚠️⚠️**Note**: For testing datasets with higher resolution than the one used for training, scaling down the images by a factor of 2 (i.e., scale=0.5) and then upsampling the predictions back to the original resolution speeds up inference and can improve results. Thanks to [tpy001](https://github.com/tpy001) for raising this point in the [issues](https://github.com/astra-vision/FAMix/issues/5). The scale parameters can be customized when running [Evaluation](#evaluation) by adding --scale <value>.

Backbone | Decoder | Scale   | Cityscapes   |   Mapillary    | ACDC night | ACDC snow | ACDC rain | ACDC fog
| :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: |
RN50 | DLv3+ | 1      | **48.51**  | 52.39 | 15.02 | 37.38 | **39.56** | 40.99 |  
RN50 | DLv3+ | 0.5    | 48.02 | **54.00** | **21.58** | **38.27** | 39.53 | **44.94** | 
RN101 | DLv3+ | 1     | 49.13 | 53.41 | 21.28 | **41.49** | 42.19 | 44.30 |  
RN101 | DLv3+ | 0.5   | **50.06** | **55.31** | **23.97** | 40.34 | **42.41** | **44.98** |

# Table of Content
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
  - [Trained models](#trained-models)
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

## Trained models
The trained models are available [here](https://drive.google.com/drive/folders/1GzajnYyDCUL7Xl0zjjU7_n73ucuyU5SS).

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
