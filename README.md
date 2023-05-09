# PointFlowHop: Green and Interpretable Scene Flow Estimation from Consecutive Point Clouds

PointFlowHop is based on the Green Learning paradigm and decomposes 3D scene flow estimation into vehicle ego-motion compensation and object-wise motion estimation steps. 

[[arXiv](https://arxiv.org/abs/2302.14193)]

## Introduction

R-PointHop is an unsupervised learning method for registration of two point cloud objects. It derives point features from training data statistics in a hierarchical feedforward manner without end-to-end optimization. The features are used to find point correspondences which in turn lead to estimating the 3D transformation. More technical details can be found in our paper. 

In this repository, we release the code for training R-PointHop method and evaluating on a given pair of point cloud objects.

## Packages

The code has been developed and tested in Python 3.6. The following packages need to be installed.

```
h5py
numpy
scipy
sklearn
open3d
```

## Training

Train the model on all 40 classes of ModelNet40 dataset

```
python train.py --first_20 False
```

Train the model on first 20 classes of ModelNet40 dataset

```
python train.py --first_20 True
```

User can specify other parameters like number of points in each hop, neighborhood size and energy threshold, else default parameters will be used.

## Registration 

```
python test.py --source ./data/source_0.ply --target ./data/target_0.ply
```

A set of sample source and target point cloud objects are present in the [data](https://github.com/pranavkdm/R-PointHop/tree/main/data) folder which can be used for testing. Replace source_0 and target_0 with your choice of souce and target.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{kadam2023pointflowhop,
  title={PointFlowHop: Green and Interpretable Scene Flow Estimation from Consecutive Point Clouds},
  author={Kadam, Pranav and Gu, Jiahao and Liu, Shan and Kuo, C-C Jay},
  journal={arXiv preprint arXiv:2302.14193},
  year={2023}
}
```