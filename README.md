## pipe EFPN implementation based on detectron2

#### To train:

$python frcnn_x101_efpn.py


#### Modifications from detectron2:

1. In detectron2/modeling/backbone, we modified resnet.py, ftt.py and fpn.py to create the EFPN and FTT module

2. slight modifications to fix import paths, cuda compatibility...etc, including in detectron/engine/defaults.py, detectron/layers/wrappers.py


#### Credits:

EFPN original paper: C. Deng, M. Wang, L. Liu, and Y. Liu.  Extended feature pyramid network for small objectdetection.CVPR, 2020.

(https://arxiv.org/pdf/2003.07021v1.pdf)

Detectron2: Yuxin Wu, Alexander Kirillov, Francisco Massa and Wan-Yen Lo, & Ross Girshick. (2019). Detectron2. https://github.com/facebookresearch/detectron2.

Gene chou, EFPN-detectron2 :   https://github.com/gene-chou/EFPN-detectron2



#### Network:
P_2s=E_t (E_f (P_3 ‖P_4 ‖ P_5)↑_2x+E_f (P_2 ))+E_f (P_3 ‖P_4 ‖ P_5)↑_2x

whereE_t (*)denotes texture extractor component, E_f (*) denotes content extractor component, ↑_2x here denotes double upscaling by sub-pixel convolution, and || denotes feature concatenation

![image](https://user-images.githubusercontent.com/30168759/125766383-51ef36d6-0ff1-4e9e-92a8-2ff401781c32.png)

