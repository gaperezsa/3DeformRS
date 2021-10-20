import pandas as pd
import numpy as np
import matplotlib
import argparse
import math
import matplotlib.pyplot as plt
import multiprocessing
'''
This program will be able to retrieve the graph for a list of models, running some kind of deformation and given the sigmas interested in.
This is all asuming that when experiments where run, the convention {model}{deformation}{dataset spec if needed}{sigma} was followed
examples: pointnet2GaussianNoise0.02 , dgcnnShearing0.4 , dgcnnAffineModelnet10_0.005
'''
#change these as needed for current query
models=["pointnet","pointnet2","dgcnn","curvenet"]
deformation="Rotation"
usingModelnet10 = False
sigmas = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]
counter = 1
base_path = "../output/certify/"
common_end = "/certification_chunk_1out_of1.csv"
current_experiment = ""

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--parallel', action='store_true', default=False, help='certify with uniform distribution')
parser.add_argument('--envelope', action='store_true', default=False, help='certify with uniform distribution')

args = parser.parse_args()

def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs])

def allYvals(dfsAndXdomain):
    df,Xdomain = dfsAndXdomain
    return [( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/totalRows for i in Xdomain]

#this numbers based on the fact that ,at max, 4 models will be requested
rows = math.ceil(len(models)/2)
columns = 2 if len(models)>=2 else 1

for model in models:
    plt.subplot(rows, columns, counter)
    try:
        if not usingModelnet10:
            current_experiments = [model+deformation+str(sigma) for sigma in sigmas]
        else:
            current_experiments = [model+deformation+"Modelnet10_"+str(sigma) for sigma in sigmas]
        csvPaths = [base_path+current_experiment+common_end for current_experiment in current_experiments]
        dfs = [pd.read_csv(csvPath,skiprows=1) for csvPath in csvPaths] #first row is the command used, not needed here
        totalRows = dfs[0].count()[0]
        Xdomains = [np.linspace(0,df["radius"].max(),num=totalRows) for df in dfs]
        if args.parallel:
            pool_obj = multiprocessing.Pool()
            Yvalues = pool_obj.map(allYvals,zip(dfs,Xdomains))
        else:
            Yvalues = [[( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/totalRows for i in Xdomain] for df,Xdomain in zip(dfs,Xdomains)]
        print(model+deformation+' \u03C3='+str(sigmas)+'\n')
        for Xdomain,sigma,Yvalue in zip(Xdomains,sigmas,Yvalues):
            if args.envelope:
                plt.plot(Xdomain.tolist(), Yvalue,'--',label='\u03C3='+str(sigma))
            else:
                plt.plot(Xdomain.tolist(), Yvalue,label='\u03C3='+str(sigma))
        if args.envelope:
            print('calculating envelope...')
            maxRadius = max([ Xdomain[-1] for Xdomain in Xdomains ])
            EnvelopeXdomain = np.linspace(0,maxRadius,num=totalRows*len(sigmas))
            if args.parallel:
                second_pool_obj = multiprocessing.Pool()
                EnvelopeYvalues = second_pool_obj.map(checkDfs,EnvelopeXdomain)
            else:
                EnvelopeYvalues = [max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs]) for domainValue in EnvelopeXdomain]
            plt.plot(EnvelopeXdomain.tolist(), EnvelopeYvalues,label='envelope')
            print('done!\n')
    except:
        print("unable to display {}".format(current_experiment))
        import sys
        sys.exit()
    # Settings
    plt.title(model+" "+deformation+" certification")
    plt.xlabel('certification radius')
    plt.ylabel('certified accuracy')
    plt.ylim([0,1])
    plt.legend()
    plt.grid()
    counter+=1

plt.show()
