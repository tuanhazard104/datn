# Applying PVT to Semantic Segmentation

Here, we take [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as an example, applying PVTv2 to SemanticFPN.

For details see [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). 

If you use this code for a paper please cite:

```
@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Usage

Install MMSegmentation.


## Data preparation

First, prepare ADE20K according to the guidelines in MMSegmentation.

Then, download the weights pretrained on ImageNet at [here](https://drive.google.com/drive/folders/1hCaKNrlMF6ut0b36SedPRNC_434R8VVa?usp=sharing), and put them in a folder `pretrained/`


## Results and models

|    Backbone     | Iters | mIoU | Config | Download  |
| :-------------: | :-----: | :------: | :------------: | :----: |
|    PVTv2-B0 + Semantic FPN    | 40K | 37.2 |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b0_ade20k_40k.py)  |  |
|    PVTv2-B1 + Semantic FPN  | 40K |  42.5  |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b1_ade20k_40k.py)  |  |
|    PVTv2-B2 + Semantic FPN  | 40K | 45.2   |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b2_ade20k_40k.py)  |  |
|    PVTv2-B3 + Semantic FPN   | 40K |  47.3  |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b3_ade20k_40k.py)  |  |
|    PVTv2-B4 + Semantic FPN   | 40K | 47.9    |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b4_ade20k_40k.py)  |  |
|    PVTv2-B5 + Semantic FPN   | 40K |  48.7  |  [config](https://github.com/whai362/PVTv2-Seg/blob/master/configs/sem_fpn/PVTv2/fpn_pvtv2_b5_ade20k_40k.py)  |  |

## Evaluation
To evaluate PVTv2-B2 + SemFPN on a single node with 8 gpus run:
```
dist_test.sh configs/sem_fpn/PVT/fpn_pvtv2_b2_ade20k_40k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train PVTv2-B2 + SemFPN on a single node with 8 gpus run:

```
dist_train.sh configs/sem_fpn/PVT/fpn_pvtv2_b2_ade20k_40k.py 8
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
