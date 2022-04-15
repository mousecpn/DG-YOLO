# DG-YOLO

This repository contains the code (in PyTorch) for the paper:

[TOWARDS DOMAIN GENERALIZATION IN UNDERWATER OBJECT DETECTION](https://arxiv.org/abs/2004.06333)
Hong Liu, Pinhao Song, Runwei Ding

The code of this repository is based on [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

### Dependencies

- Python 3.6
- PyTorch 1.0
- CUDA9.0 and cuDNN
- numpy
- tqdm
- tensorboardX



### Installation

**Download pretrained weights**

```
$ cd weights/
$ bash download_weights.sh
```

**Download Datasets**

URPC2019: https://drive.google.com/open?id=1n8Rpgx3xF84HO6PXpfPrRtTMtVSuOaBs

Synthetic URPC2019: https://drive.google.com/open?id=1FzIuZJuCHna4Dn_FLBeR5IFztCJBJ6VD

checkpoint: https://drive.google.com/file/d/1n3e9R1zeJjOtNSMpNY0cVsfHqucmTOVe/view?usp=sharing

After downloading all datasets, create URPC2019 document.

```
$ cd data
$ mkdir URPC2019
```

 It is recommended to symlink the dataset root to `$DG-YOLO/data/URPC2019`.

```
DG-YOLO
├── data
│   ├── URPC2019
│   │   ├── type1
│   │   ├── type2
│   │   ├── type3
│   │   ├── type4
│   │   ├── type5
│   │   ├── type7
│   │   ├── val_type1
│   │   ├── val_type2
│   │   ├── val_type3
│   │   ├── val_type4
│   │   ├── val_type5
│   │   ├── val_type6
│   │   ├── val_type7
│   │   ├── val_type8
│   │   ├── train2017
│   │   ├── val2017
```

###  

### Train

```
$ python DG_train.py --pretrained_weights ./weights/darknet53.conv.74 --batch_size 8
```



### Test

Test in original validation set

```
$ python test.py --weights_path <path/to/checkpoints> --batch_size 32
```

Test in type8 validation set

```
$ python test.py --weights_path <path/to/checkpoints> --batch_size 32 --augment True
```

### Citation

```
@inproceedings{liu2020towards,
  title={Towards domain generalization in underwater object detection},
  author={Liu, Hong and Song, Pinhao and Ding, Runwei},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={1971--1975},
  year={2020},
  organization={IEEE}
}
```
