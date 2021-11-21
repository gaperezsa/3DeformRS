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
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--query_radius', type=float, default=0.1,help='radius of interest')
args = parser.parse_args()

#change these as needed for current query
models=["64pointnet"]#,"pointnet2","dgcnn","curvenet"]
deformation="RotationX"
usingModelnet10 = False
sigmas = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4]#[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
base_path = "../output/3DcertifyComparison/"
common_end = "/certification_chunk_1out_of1.csv"



def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs])


for model in models:
    try:
        if not usingModelnet10:
            current_experiments = [model+deformation+str(sigma) for sigma in sigmas]
        else:
            current_experiments = [model+deformation+"Modelnet10_"+str(sigma) for sigma in sigmas]
        csvPaths = [base_path+current_experiment+common_end for current_experiment in current_experiments]
        dfs = [pd.read_csv(csvPath,skiprows=1) for csvPath in csvPaths] #first row is the command used, not needed here
        totalRows = dfs[0].count()[0]
        answer = max([( (df["correct"] == 1) & (df["radius"] >= args.query_radius) ).sum()/totalRows for df in dfs])
        print("certification accuracy for {} {} at radius {} is {}".format(model,deformation,args.query_radius,answer))
    except:
        print("unable to display for {}".format(model))
