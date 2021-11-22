# 3DeformRS
Official implementation of "3DeformRS: Certifying Spatial Deformations on Point Clouds"

![3DeformRS](./pull_pc.png)

## Requirements (tested on)
- Python = 3.8
- PyTorch = 1.9 , cuda = 11.1
- Pytorch-geometric = 1.7
- scipy = 1.7

## Setup
Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/), change directory to ```3D-RS-PointCloudCertifying/ ``` and execute:
```
conda env create -f environment.yml
conda activate CertifyingPointclouds
```


## Data

Two major datasets were used, ModelNet40 and ScanObjectNN


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



## Training and Testing

Pre-trained models from us as well as original authors of the assessed DNN's can be found in the corresponding folder for every module:
* ```3D-RS-PointCloudCertifying/CurveNet/checkpoints/``` for CurveNet
* ```3D-RS-PointCloudCertifying/Pointnet/trainedModels``` for PointNet
* ```3D-RS-PointCloudCertifying/Pointnet2andDGCNN/trainedModels``` for PointNet++ and DGCNN

The following examples use ScanObjectNN, be aware that to use ModelNet40, value "modelnet40" must be passed instead of modelnet40 under the --dataset flag (also recommended to put a coherent experiment_name but that is up to the user).

**NOTE:** all following commands assume the current working directory is ```3D-RS-PointCloudCertifying/``` when beggining.

### CurveNet

**Training**

```
cd CurveNet/core/
python3 main_cls.py --exp_name curvenetScanObjectNNBaseLine --dataset scanobjectnn --epochs 100 --num_workers 10
```
**Testing**
```
python3 main_cls.py --eval True --exp_name cuvenetScanObjectNNBaseLine --dataset scanobjectnn --num_workers 10 --model_path [your_own_path]/3D-RS-PointCloudCertifying/CurveNet/checkpoints/modelNet40BaseLine/models/model.t7
```

### PointNet

**Training**

```
cd Pointnet/
python3 train.py --experiment_name pointnetScanObjectNNBaseLine --dataset scanobjectnn --epochs 100 --num_workers 10 --rotation none
```

**Testing**

```
python3 test.py --experiment_name pointnetScanObjectNNBaseLine --dataset scanobjectnn --num_workers 10
```

### PointNet++

**Training**

```
cd Pointnet2andDGCNN/Trainers/
python3 pointnet2Train.py --experiment_name pointnet2ScanObjectNNBaseline --dataset scanobjectnn --epochs 100
```

**Testing**

```
python3 pointnet2Test.py --experiment_name pointnet2ScanObjectNNBaseline --dataset scanobjectnn
```

### DGCNN

**Training**

```
cd Pointnet2andDGCNN/Trainers/
python3 dgcnnTrain.py --experiment_name dgcnnScanObjectNNBaseline --dataset scanobjectnn --epochs 100
```

**Testing**

```
python3 dgcnnTest.py --experiment_name dgcnnScanObjectNNBaseline --dataset scanobjectnn
```

## Certify

After having trained a model, Certify.py can receive a path to the model, name of the network, in which dataset it was trained, the certified method or perturbation to be certified against, the sigma noise hyperparameter and the name of this experiment.

From this point on, we will follow the user case of certifying a **Pointnet++ instance against rotation Z on ScanObjectNN** and producing figures such as the ones in the paper. However, the idea is the same for any other network, deformation or dataset that one may need.

dataset_choices = ['modelnet40','scanobjectnn']

model_choices = ['pointnet2','dgcnn','curvenet','pointnet']

certification_method_choices = ['RotationX','RotationY','RotationZ','RotationXZ','RotationXYZ','Translation','Shearing','Tapering','Twisting','Squeezing','Stretching','GaussianNoise','Affine','AffineNoTranslation']

Beware, perturbation choices such as squeezing or stretching are ill defined for interpreting their certified radius and so, they were taken out of the official paper.

### Run the experiments

From the ```3D-RS-PointCloudCertifying/``` directory:

```
python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.05 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.05

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.1 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.1

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.15 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.15

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.2 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.2

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.25 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.25

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.3 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.3

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.35 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.35

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.4 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.4

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.45 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.45

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.5 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.5

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.55 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.55

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.6 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.6

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.65 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.65

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.7 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.7

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.75 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.75

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.8 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.8

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.85 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.85

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.9 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.9

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 0.95 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ0.95

python3 Certify.py --model pointnet2 --dataset scanobjectnn --base_classifier_path Pointent2andDGCNN/trainedModels/pointnet2ScanObjectNNBaseline/FinalModel.pth.tar --sigma 1 --certify_method rotationZ --uniform --experiment_name scanobjectnnPointnet2RotationZ1

```

Each of these experiments will take between 30 min to 2 hours depending on your computer's specifications.

There should be one folder per experiment under ```3D-RS-PointCloudCertifying/output/certify/scanobjectnn/RotationZ/``` 

inside each of these folder, a .txt and a .csv version of the results can be found with the columns:

* Correct label
* prediction of the smoothed classifier
* certified radius
* boolean value if correct
* time taken

### Analyse results, Make the corresponding graphics

Under the Analysis folder, two programs of interest may be found: 

1. getGraph.py
2. getEnvelope.py

Naming convention is required for this programs to work, that is, certification experiments were named: *[Model][Deformation][Noise Hyperparameter]

some examples:

* scanobjectnnPointnet2RotationZ0.95
* modelnet40dgcnnRotationXZ0.1
* curvenetGaussianNoise0.06
* pointnetTranslation0.25



It is **strongly** recommended to go into said files and change the default values and settings for producing theses graphs according to your needed query. By default, when running Certify.py, the directory ```3D-RS-PointCloudCertifying/ouput/certify/[dataset]/[deformation]/``` should now exist per deformation and with all results from one common deformation under it.

#### getGraph.py

To reproduce the different sigmas graph for rotation Z on ScanObjectNN with pointnet++:

set models = ['Pointnet2']

set deformation = "RotationZ"

set base_path = "../output/certify/scanobjectnn/RotationZ/"

set save_path = '/[path to where you want the graphs to be saved]/'


after this:

```
cd Analysis/
python3 getGraph.py
```

flag --parallel is recommended if multi-core computation is available.

flag --envelope will output the same graph but with every sigma curve dashed and a thicker envelope curve.

#### getEnvelope.py

This program was created with the intent to directly generate all deformations envelope comparisons, as seen on the paper, rather than setting one model per graph and showing the noise hyperparameters they depend on like the ones produced by: getGraph.py

To only reproduce the envelopes graph for rotation Z on ScanObjectNN :

set models = ['Pointnet','Pointnet2','Dgcnn','Curvenet']

set base_path = "../output/certify/scanobjectnn/"

set save_path = '/[path to where you want the graphs to be saved]/'


after this:

```
cd Analysis/
python3 getGraph.py --dataset scanobjectnn --deformation RotationZ
```

flag --parallel is recommended if multi-core computation is available.

Not passing the --deformation flag or passing the value "all" are equivalent and all deformations will be searched for and attempted to get the envelope out of.