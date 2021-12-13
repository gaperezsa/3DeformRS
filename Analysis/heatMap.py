import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import matplotlib
import argparse
import math
import matplotlib.pyplot as plt
import multiprocessing
import glob
from sklearn import metrics
import seaborn as sns
import csv

data_origins=["precalculated_envelope_csv","manual"]

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--dataset', type=str, default='modelnet40' ,help='over which dataset')
parser.add_argument('--data_origin', type=str, default='precalculated_envelope_csv' ,choices=data_origins,help='over which dataset')

args = parser.parse_args()


#change these as needed for current query

if args.data_origin == "precalculated_envelope_csv":


    

    if args.dataset == "modelnet40":
        model="pointnet2"
        augmentations=["Affine0.1","GaussianNoise0.03","RotationZ0.2","RotationXYZ0.1","Translation0.2","Twisting0.8"] #just used for directory name matching
        deformations=["Affine","GaussianNoise","RotationZ","RotationXYZ","Translation","Twisting"]
        base_path = "../output/certify/modelnet40/"
        save_path = '/home/santamgp/Downloads/CVPRGraphics/TestingREADME/'
    elif args.dataset == "scanobjectnn":
        models=["Pointnet","Pointnet2","Dgcnn","Curvenet"]
        deformations=["Affine","GaussianNoise","RotationXYZ","Translation","Twisting"]
        base_path = "../output/certify/scanobjectnn/"
        save_path = '/home/santamgp/Downloads/CVPRGraphics/TestingREADME/'

    data = np.random.randn(len(augmentations), len(deformations))
    for augmentationIndex in range(len(augmentations)):
        for deformationIndex in range(len(deformations)):
            preCalculatedData = glob.glob(f"{base_path}/*{deformations[deformationIndex]}/*{augmentations[augmentationIndex]}{model}{deformations[deformationIndex]}EnvelopeValues.csv")
            notAugmentedData = glob.glob(f"{base_path}/*{deformations[deformationIndex]}/{model}{deformations[deformationIndex]}EnvelopeValues.csv")
            dfs = pd.read_csv(preCalculatedData[0],header=None)
            basedfs = pd.read_csv(notAugmentedData[0],header=None)
            data[augmentationIndex][deformationIndex] = ( metrics.auc(dfs[0].tolist(), dfs[1].tolist()) - metrics.auc(basedfs[0].tolist(), basedfs[1].tolist()) )/ metrics.auc(basedfs[0].tolist(), basedfs[1].tolist())

elif args.data_origin == "manual":
    data = np.random.rand(10, 12)

heatmap = sns.heatmap(data, center=0,vmax=1,annot=True,cmap='coolwarm',linewidth=0.2,cbar_kws={'label': 'Relative improvement of ACR'},xticklabels=deformations, yticklabels=augmentations)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30)
plt.title(model, fontsize=30)
plt.xlabel("Certified Against")
plt.ylabel("Augmentation method")

plt.show()