# PointFlowHop: Green and Interpretable Scene Flow Estimation from Consecutive Point Clouds

PointFlowHop is based on the Green Learning paradigm and decomposes 3D scene flow estimation into vehicle ego-motion compensation and object-wise motion estimation steps. 

[[arXiv](https://arxiv.org/abs/2302.14193)]

## Overview

PointFlowHop takes two
consecutive point clouds and determines the 3D flow vectors for every point in the first point cloud. PointFlowHop
decomposes the scene flow estimation task into a set of
subtasks, including ego-motion compensation, object association and object-wise motion estimation. It follows the
green learning (GL) pipeline and adopts the feedforward
data processing path. As a result, its underlying mechanism is more transparent than deep-learning (DL) solutions
based on end-to-end optimization of network parameters.
We conduct experiments on the stereoKITTI and the Argoverse LiDAR point cloud datasets and demonstrate that
PointFlowHop outperforms deep-learning methods with a
small model size and less training time. Furthermore, we
compare the Floating Point Operations (FLOPs) required
by PointFlowHop and other learning-based methods in inference, and show its big savings in computational complexity.





## Packages

The code has been developed and tested in Python. The following packages need to be installed.

```
h5py
numpy
scipy
sklearn
open3d
pyntcloud
```

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