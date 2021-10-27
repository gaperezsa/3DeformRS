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
parser.add_argument('--parallel', action='store_true', default=False, help='add flag to use parallel computation of the graphs')
parser.add_argument('--envelope', action='store_true', default=False, help='add flag to make the envelope curve appear solid and the other as dashlines underneath it')
parser.add_argument('--hypervolume', action='store_true', default=False, help='add flag to make the X axis in terms of hypervolume certified')
args = parser.parse_args()

#change these as needed for current query
models=["pointnet","pointnet2","dgcnn","curvenet"]
deformation="RotationZ"
usingModelnet10 = False
sigmas = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
counter = 1
base_path = "../output/certify/ZRotations/"
common_end = "/certification_chunk_1out_of1.csv"
current_experiment = ""


#function used if parallel computation asked
def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs])

#function used if parallel computation asked
def allYvals(dfsAndXdomain):
    df,Xdomain = dfsAndXdomain
    return [( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/totalRows for i in Xdomain]

#calculate hypervolume based on radius if deformation is rotation xyz
def hyperVolofRotXYZ(radius):
    transformedDomain = (4/3)*np.power(radius,3) #volume of a 3-ball in L1 norm (octahedron)
    return transformedDomain

#calculate hypervolume based on radius if deformation is rotation xyz
def hyperVolofRotXZ(radius):
    transformedDomain = np.square(np.sqrt(2)*radius) #area of a 2-ball in L1 norm (rhombus)
    return transformedDomain

#this numbers based on the fact that ,at max, 4 models will be requested
rows = math.ceil(len(models)/2)
columns = 2 if len(models)>=2 else 1

def DomainTransformer(deformation,radius):
    switcher = {
        "Rotation"   : hyperVolofRotXYZ,
        "RotationXYZ": hyperVolofRotXYZ,
        "RotationXZ" : hyperVolofRotXZ
    }
    return switcher.get(deformation,"not a valid deforamtion with defined hypervolume")(radius)

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
        Xdomains = [np.append(np.linspace(0,df["radius"].max(),num=totalRows) ,df["radius"].max() + (df["radius"].max()/(totalRows-1)) ) for df in dfs]
        if args.parallel:
            pool_obj = multiprocessing.Pool()
            Yvalues = pool_obj.map(allYvals,zip(dfs,Xdomains))
        else:
            Yvalues = [[( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/totalRows for i in Xdomain] for df,Xdomain in zip(dfs,Xdomains)]
        print(model+deformation+' \u03C3='+str(sigmas)+'\n')
        showLegendCounter = 0
        for Xdomain,sigma,Yvalue in zip(Xdomains,sigmas,Yvalues):
            if (args.hypervolume):
                plottingDomain = DomainTransformer(deformation,Xdomain)
            else:
                plottingDomain = Xdomain

            label = '\u03C3='+str(sigma) if showLegendCounter%2==0 else '_\u03C3='+str(sigma) #those labels starting with _ are ignored by the automatic legend
            showLegendCounter += 1

            if args.envelope:
                plt.plot(plottingDomain.tolist(), Yvalue,'--',label=label)
            else:
                plt.plot(plottingDomain.tolist(), Yvalue,label=label)
        if args.envelope:
            print('calculating envelope...')
            maxRadius = max([ Xdomain[-1] for Xdomain in Xdomains ])
            EnvelopeXdomain = np.append(np.linspace(0,maxRadius,num=totalRows*len(sigmas)) ,maxRadius + maxRadius/(totalRows-1))

            if args.parallel:
                second_pool_obj = multiprocessing.Pool()
                EnvelopeYvalues = second_pool_obj.map(checkDfs,EnvelopeXdomain)
            else:
                EnvelopeYvalues = [max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/totalRows for df in dfs]) for domainValue in EnvelopeXdomain]
            
            if (args.hypervolume):
                plottingDomain = DomainTransformer(deformation,EnvelopeXdomain)
            else:
                plottingDomain = EnvelopeXdomain
            plt.plot(plottingDomain.tolist(), EnvelopeYvalues,label='envelope')
            print('done!\n')
        
        #draw other papers points
        if (model == "pointnet" and (deformation == "RotationXYZ" or deformation == "Rotation")  and args.hypervolume):
            plt.plot(np.power(np.pi,3)/5832,0.587,'ro',label='3D certify')
            plt.plot(np.power(np.pi,3)/91125,0.728,'ro')
        elif (model == "pointnet" and deformation == "RotationXZ" and args.hypervolume):
            plt.plot(np.square(np.pi)/81,0.739,'ro',label='3D certify')
            plt.plot(np.square(np.pi)/324,0.891,'ro') 
    except:
        print("unable to display {}".format(current_experiment))
    # Settings
    plt.title(model+" "+deformation+" certification")
    if (args.hypervolume):
        plt.xlabel('certified hypervolume')
    else:
        plt.xlabel('certification radius')
    
    plt.ylabel('certified accuracy')
    plt.ylim([0,1])
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid()
    counter+=1

plt.show()
