# 3DeformRS
Official implementation of "3DeformRS: Certifying Spatial Deformations on Point Clouds",
Research done as VSRP in King Abullah's University of Science and Technology by Gabriel Pérez S

![3DeformRS](./pull_pc.png)

## Requirements (tested on)
- Python = 3.8
- PyTorch = 1.9 , cuda = 11.1
- Pytorch-geometric = 1.7
- scipy = 1.7

## Data

two major datasets were used, ModelNet40 and ScanObjectNN


### ModelNet40
At your first run, any of the training programs will automatically download the corresponding version of ModelNet40 unless already there. Implementation-wise, there are 3 "different" versions on the dataset. 


#### CurveNet
Uses the ply_hdf5_2048 version which can be manually downloaded from [offical data](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and then unzipped directly under the Data folder.

#### PointNet
Downloads the torch geometric version of the dataset, however, uses hand-made pre-transforms which make it incompatible with the vanilla torch geometric version of the dataset. namely, it applies transformers.SamplePoints(num=4096) transformers.FarthestPoints(num_points=1024). pre-transforms made by this implementation were kept in order to be consistent when comparing against [3D Certify](https://github.com/eth-sri/3dcertify)

#### PointNet++ and DGCNN
Both of these implementations are taken from the torch geometric examples and, because of this, uses the torch geometric integrated version of the dataset. 

### ScanObject
The ScanObject dataset can be obtained for reserach purposes following the correct process as stated in their public [page](https://hkust-vgd.github.io/scanobjectnn/). That is, writing an email to <mikacuy@gmail.com>, accepting terms of use and downloading. Once obtained please **unzip and place directly on the Data folder**. If you need this dataset in a different folder, flag --data_dir can be passed with the absolute path to it.

**NOTE:** all following commands assume the current working directory is ```3D-RS-PointCloudCertifying/``` when beggining.

## Training and Testing

pre-trained models from us as well as original authors of the assessed DNN's can be found in the corresponding folder for every module:
* ```3D-RS-PointCloudCertifying/CurveNet/checkpoints/``` for CurveNet
* ```3D-RS-PointCloudCertifying/Pointnet/trainedModels``` for PointNet
* ```3D-RS-PointCloudCertifying/Pointnet2andDGCNN/trainedModels``` for PointNet++ and DGCNN

### CurveNet
Uses the ply_hdf5_2048 version which can be manually downloaded from [offical data](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and then unzipped directly under the Data folder.

### PointNet
Downloads the torch geometric version of the dataset, however, uses hand-made pre-transforms which make it incompatible with the vanilla torch geometric version of the dataset. namely, it applies transformers.SamplePoints(num=4096) transformers.FarthestPoints(num_points=1024). pre-transforms made by this implementation were kept in order to be consistent when comparing against [3D Certify](https://github.com/eth-sri/3dcertify)

### PointNet++ and DGCNN
Both of these implementations are taken from the torch geometric examples and, because of this, uses the torch geometric integrated version of the dataset. 


```
./start_cls.sh
```

python3 train.py --experiment_name debugging --dataset modelnet40 --epochs 2 --num_workers 10 --rotation none
