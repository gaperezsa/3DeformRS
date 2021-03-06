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
sns.set_theme()
sns.set_style("darkgrid")
'''
This program will be able to retrieve the graph for a list of models, running some kind of deformation and given the sigmas interested in.
This is all asuming that when experiments where run, the convention {model}{deformation}{sigma} was followed
examples: pointnet2GaussianNoise0.02 , dgcnnShearing0.4 , dgcnnAffine0.005
'''
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--parallel', action='store_true', default=False, help='add flag to use parallel computation of the graphs')
parser.add_argument('--deformation', type=str,default='all', help='which deformation')
parser.add_argument('--dataset', type=str, default='modelnet40' ,help='unused, just redirect correctly the base path')
args = parser.parse_args()


#change these as needed for current query
if args.dataset == "modelnet40":
    models=["pointnet","pointnet2","dgcnn","curvenet"]
    base_path = "../output/certify/"
    save_path = '/home/santamgp/Downloads/CVPRGraphics/TestingREADME/'
elif args.dataset == "scanobjectnn":
    models=["Pointnet","Pointnet2","Dgcnn","Curvenet"]
    base_path = "../output/certify/scanobjectnn/"
    save_path = '/home/santamgp/Downloads/CVPRGraphics/TestingREADME/'

deformation= args.deformation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})
current_experiment = ""

xlabels={"Affine":r'$\sqrt{a^2+b^2+c^2+d^2+e^2+f^2+g^2+h^2+i^2+j^2+k^2+l^2}$',
        "AffineNoTranslation":r'$\sqrt{a^2+b^2+c^2+d^2+e^2+f^2+g^2+h^2+i^2}$',
        "GaussianNoise":r'$||\phi||_2$',
        "RotationZ":r'$|\gamma|$',
        "RotationXZ":r'$|\beta|+|\gamma|$',
        "Rotation":r'$|\alpha|+|\beta|+|\gamma|$',
        "RotationXYZ":r'$|\alpha|+|\beta|+|\gamma|$',
        "Shearing":r'$\sqrt{a^2+b^2}$',
        "Squeezing":r'$|\bar{k}|$',
        "Stretching":r'$|\bar{k}|$',
        "Tapering":r'$\sqrt{a^2+b^2}$',
        "Translation":r'$\sqrt{tx^2+ty^2+tz^2}$',
        "Twisting":r'$|\gamma|$'}

abreviations={
        "pointnet": "PointNet",
        "Pointnet": "PointNet",
        "pointnet2":"PointNet++",
        "Pointnet2":"PointNet++",
        "dgcnn":"DGCNN",
        "Dgcnn":"DGCNN",
        "curvenet":"CurveNet",
        "Curvenet":"CurveNet"
        }

deformationTitles={"Affine":'Affine',
        "AffineNoTranslation":"Affine (NT)",
        "GaussianNoise":"Gaussian Noise",
        "RotationZ":"Rotation (Z)",
        "RotationXZ":"Rotation (XZ)",
        "Rotation":"Rotation (XYZ)",
        "RotationXYZ":"Rotation (XYZ)",
        "Shearing":"Shearing",
        "Squeezing":"Squeezing",
        "Stretching":"Stretching",
        "Tapering":"Tapering",
        "Translation":"Translation",
        "Twisting":"Twisting"}
        

if (deformation == 'all'):
    deformations = deformationTitles.keys()
else:
    deformations = [deformation]


def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/(df.count()[0]) for df in dfs])

for deformation in deformations:
    for model in models:
        try:
            preCalculatedData = glob.glob(f"{base_path}/*{deformation}/*{model}{deformation}EnvelopeValues.csv")
            dfs = pd.read_csv(preCalculatedData[0],header=None)
            sns.lineplot(x=dfs[0].tolist(), y=dfs[1].tolist(),label=f"{abreviations[model]}")
        except:
            print(f'no precalculated data found')
            try:
                csvPaths = glob.glob(f"{base_path}/*{deformation}/*{model}{deformation}*/*.csv")
                dfs = [pd.read_csv(csvPath,skiprows=1) for csvPath in csvPaths] #first row is the command used, not needed here
                print(f"samples certified per sigma for {model}")
                totalRows = max([df.count()[0] for df in dfs])
                print('calculating envelope...')
                maxRadius = max([ max(df["radius"]) for df in dfs ])
                EnvelopeXdomain = np.append(np.linspace(0,maxRadius,num=totalRows*len(csvPaths)) , maxRadius + (maxRadius/(totalRows*len(csvPaths)-1)))
                if args.parallel:
                    second_pool_obj = multiprocessing.Pool()
                    EnvelopeYvalues = second_pool_obj.map(checkDfs,EnvelopeXdomain)
                else:
                    EnvelopeYvalues = [max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/(df.count()[0]) for df in dfs]) for domainValue in EnvelopeXdomain]
                #sns.lineplot(x=EnvelopeXdomain.tolist(), y=EnvelopeYvalues,label=f"{abreviations[model]} ACR={metrics.auc(EnvelopeXdomain.tolist(), EnvelopeYvalues):.2f}")
                sns.lineplot(x=EnvelopeXdomain.tolist(), y=EnvelopeYvalues,label=f"{abreviations[model]}" )
                print('done!\n')
                with open(f'{base_path}/{deformation}/{model}{deformation}EnvelopeValues.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(EnvelopeXdomain.tolist(), EnvelopeYvalues))
            except:
                print("unable to display {}".format(current_experiment))
    
    # Settings

    #plt.title(deformation, fontsize=30)
    plt.text(.5,.9,deformationTitles[deformation],
            horizontalalignment='center',
            transform=plt.gca().transAxes,
            fontsize=30)
    if (deformation == 'Affine'):
        ftsz=15
    elif (deformation == 'AffineNoTranslation'):
        ftsz=17
    else :
        ftsz=20

    plt.xlabel(xlabels[deformation], fontsize=ftsz)
    if (deformation == 'RotationZ'or deformation == 'Twisting'):
        plt.ylabel('Certified Accuracy', fontsize=20)
        plt.yticks(fontsize=20)
    else:
        plt.gca().axes.yaxis.set_ticklabels([])

    plt.xticks(fontsize=16)
    plt.ylim([0,1])
    plt.gca().set_xlim(left=0)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1),framealpha=0.5, fontsize=12)#, ncol=len(models))#
    #plt.legend(loc='lower left', bbox_to_anchor=(0, 0),framealpha=0.5, fontsize=12)
    #plt.grid()
    plt.savefig(f"{save_path}{deformation}Envelope.png",bbox_inches='tight')
    plt.savefig(f"{save_path}{deformation}Envelope.pdf",bbox_inches='tight')
    plt.savefig(f"{save_path}{deformation}Envelope.eps",bbox_inches='tight')
    #plt.show()
    plt.clf()