import pandas as pd
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
'''
This program will be able to retrieve the evelope graph for a list of models, running some kind of deformation and given the sigmas interested in.
this is all asuming that when experiments where run, the convention {model}{deformation}{dataset spec if needed}{sigma} was followed
examples: pointnet2GaussianNoise0.02 , dgcnnShearing0.4 , dgcnnAffineModelnet10_0.005
'''
#change these as needed for current query
models=["pointnet","pointnet2","dgcnn","curvenet"]
deformation="Rotation"
usingModelnet10 = False
withDashLines = True
sigmas = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]

samples = 300
localSamples = 80
counter = 1
base_path = "../output/certify/"
common_end = "/certification_chunk_1out_of1.csv"
current_experiment = ""


def getCertifiedAccuracies(csvPath, Xdomain):
    try:
        df = pd.read_csv(csvPath,skiprows=1) #first row is the command used, not needed here
        totalRows = df.count()[0]
        Yvalues = [df.loc[ (df["correct"]==1) & (df["radius"] >= currentThreshold) ].count()[0] / totalRows for currentThreshold in Xdomain]
        return Yvalues
    except:
        return np.zeros(len(Xdomain))
    

#this numbers based on the fact that ,at max, 4 models will be requested
rows = math.ceil(len(models)/2)
columns = 2 if len(models)>=2 else 1

for model in models:
    plt.subplot(rows, columns, counter)
    maxCertifiedRadius = 0
    csv_paths = []

    #get the max certification radius
    for sigma in sigmas:
        try:
            if not usingModelnet10:
                current_experiment = model+deformation+str(sigma)
            else:
                current_experiment = model+deformation+"Modelnet10_"+str(sigma)
            csvPath = base_path+current_experiment+common_end

            #list of csv paths will be useful later
            csv_paths.append(csvPath)

            #first row is the command used, not needed here
            df = pd.read_csv(csvPath,skiprows=1) 

            totalRows = df.count()[0]
            maxCertifiedRadius = df["radius"].max() if df["radius"].max() >= maxCertifiedRadius else maxCertifiedRadius
            step = df["radius"].max() / localSamples
            if withDashLines:
                localXdomain = np.arange(localSamples+2) * step
                localXdomain[-1] = localXdomain[-2] + (np.abs(localXdomain[-2]-localXdomain[-1]))/100
                localYvalues = [df.loc[ (df["correct"]==1) & (df["radius"] >= currentThreshold) ].count()[0] / totalRows for currentThreshold in localXdomain]
                plt.plot(localXdomain.tolist(), localYvalues,'--',label='\u03C3='+str(sigma))
        except:
            print("unable to gather info for {}".format(current_experiment))
    
    #calculate the Xdomain (radii you are interested in)
    step = maxCertifiedRadius / samples
    Xdomain = np.arange(samples+2) * step

    #get the max certifiaction radius obtained for that threshold across the avaliable paths
    allYvalues = [getCertifiedAccuracies(currentPath,Xdomain) for currentPath in csv_paths]
    Yvalues = []
    print("\n"+model+"\n")
    for i in range(len(Xdomain)):
        Yvalues.append(max([u[i] for u in allYvalues ]))
        if model == 'pointnet':
            print('\nx value: {}  \ny value: {}'.format(Xdomain.tolist()[i],Yvalues[i]))
    
    plt.plot(Xdomain.tolist(), Yvalues)
    # Settings
    plt.title(model+" "+deformation+" certification")
    plt.xlabel('certification radius')
    plt.ylabel('certified accuracy')
    plt.ylim([0,1])
    if withDashLines:
        plt.legend()
    plt.grid()
    counter+=1



plt.show()
