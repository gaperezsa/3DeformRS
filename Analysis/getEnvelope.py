import pandas as pd
import numpy as np
import matplotlib
import argparse
import math
import matplotlib.pyplot as plt
import multiprocessing
from sklearn import metrics
import seaborn as sns
sns.set_theme()
sns.set_style("darkgrid")
'''
This program will be able to retrieve the graph for a list of models, running some kind of deformation and given the sigmas interested in.
This is all asuming that when experiments where run, the convention {model}{deformation}{dataset spec if needed}{sigma} was followed
examples: pointnet2GaussianNoise0.02 , dgcnnShearing0.4 , dgcnnAffineModelnet10_0.005
'''
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--parallel', action='store_true', default=False, help='add flag to use parallel computation of the graphs')
parser.add_argument('--deformation', type=str, help='add flag to use parallel computation of the graphs')
parser.add_argument('--dataset', type=str, help='add flag to use parallel computation of the graphs')
args = parser.parse_args()
# python3 getEnvelope.py --parallel --deformation Tapering --dataset ModelNet40
# python3 getEnvelope.py --parallel --deformation Translation --dataset ModelNet40
# python3 getEnvelope.py --parallel --deformation Twisting --dataset ModelNet40
# python3 getEnvelope.py --parallel --deformation XYZRotation --dataset ModelNet40


#change these as needed for current query
models=["pointnet","pointnet2","dgcnn","curvenet"]
deformation=args.deformation #"Tapering"
# usingModelnet10 = False
# sigmas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]#,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
import glob
base_path = "../output/certify/"
common_end = "/certification_chunk_1out_of1.csv"
save_path = '/home/santamgp/Downloads/CVPRGraphics/ModelNet40/'
current_experiment = ""



def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs])


for model in models:
    try:
        # if not usingModelnet10:
        #     current_experiments = [model+deformation+str(sigma) for sigma in sigmas]
        # else:
        #     current_experiments = [model+deformation+"Modelnet10_"+str(sigma) for sigma in sigmas]
        # csvPaths = [base_path+current_experiment+common_end for current_experiment in current_experiments]
        csvPaths = glob.glob(f"{base_path}/{deformation}/{model}{deformation}*/*.csv")
        dfs = [pd.read_csv(csvPath,skiprows=1) for csvPath in csvPaths] #first row is the command used, not needed here
        totalRows = dfs[0].count()[0]
        print('calculating envelope...')
        maxRadius = max([ max(df["radius"]) for df in dfs ])
        EnvelopeXdomain = np.linspace(0,maxRadius,num=totalRows*len(csvPaths))
        if args.parallel:
            second_pool_obj = multiprocessing.Pool()
            EnvelopeYvalues = second_pool_obj.map(checkDfs,EnvelopeXdomain)
        else:
            EnvelopeYvalues = [max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs]) for domainValue in EnvelopeXdomain]
        sns.lineplot(x=EnvelopeXdomain.tolist(), y=EnvelopeYvalues,label=f"{model} ACR={metrics.auc(EnvelopeXdomain.tolist(), EnvelopeYvalues):.2f}")
        print('done!\n')
    except:
        print("unable to display {}".format(current_experiment))
   
 # Settings
plt.title(model+" "+deformation+" certification")
plt.xlabel(r'$|\bar{k}|$')
plt.ylabel('certified accuracy')
plt.ylim([0,1])
plt.xlim([0,1.24])
plt.legend(loc='lower left', bbox_to_anchor=(0, 0),framealpha=0.5)
#plt.grid()
plt.savefig(f"{save_path}{deformation}.png",bbox_inches='tight')
plt.savefig(f"{save_path}{deformation}.pdf",bbox_inches='tight')
plt.savefig(f"{save_path}{deformation}.eps",bbox_inches='tight')
plt.show()
