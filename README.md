# 3D Gaussian Splatting with NuScenes Dataset

This paper presents an improved 3D Gaussian splatting (3DGS) based scene representation for autonomous driving scenarios using LiDAR-based depth regularization. The original 3DGS implementation does not explicitly account for 3D structure in its optimization and suffers in performance for sparse inputs and larger scenes, which are common in driving datasets. Infusing a depth prior can support the optimization, especially in sparse set of images. In our study, we use the NuScenes dataset to compare the depth regularized method against the original 3DGS implementation. Our results demonstrate that depth-regularized 3DGS significant improvement in depth estimates while providing comparable RGB renders. Further, qualitative assessments of novel view synthesis reveal reduced artifacts with our approach.

### Training View Rasterization
![alt text](media/training.png)

### Novel View Synthesis
Novel views rasterized using 3DGS Original (Top row) and 3DGS LiDAR Depth Regularized (Bottom row).

![Novel View Synthesis](media/novel_view.png)

### COLMAP Setup (Left) and Dense Depth Map Estimation with LiDAR Data (Right)
![alt text](media/method_highlights.png)


## Acknowledgements

1. 3D Gaussian Splatting (3DGS): https://github.com/graphdeco-inria/gaussian-splatting
2. nuscenes-devkit: https://github.com/nutonomy/nuscenes-devkit
3. Depth Regularized 3DGS: https://github.com/robot0321/DepthRegularizedGS
