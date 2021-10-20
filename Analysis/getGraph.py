import pandas as pd
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
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



samples = 80
counter = 1
base_path = "../output/certify/"
common_end = "/certification_chunk_1out_of1.csv"
current_experiment = ""

#this numbers based on the fact that ,at max, 4 models will be requested
rows = math.ceil(len(models)/2)
columns = 2 if len(models)>=2 else 1

for model in models:
    plt.subplot(rows, columns, counter)
    for sigma in sigmas:
        try:
            if not usingModelnet10:
                current_experiment = model+deformation+str(sigma)
            else:
                current_experiment = model+deformation+"Modelnet10_"+str(sigma)
            csvPath = base_path+current_experiment+common_end
            df = pd.read_csv(csvPath,skiprows=1) #first row is the command used, not needed here
            totalRows = df.count()[0]
            step = df["radius"].max() / samples
            Xdomain = np.arange(samples+2) * step
            Xdomain[-1] = Xdomain[-2] + (np.abs(Xdomain[-2]-Xdomain[-1]))/100
            Yvalues = [df.loc[ (df["correct"]==1) & (df["radius"] >= currentThreshold) ].count()[0] / totalRows for currentThreshold in Xdomain]
            print(model+deformation+' \u03C3='+str(sigma)+'\nx values: {}  \ny values: {}'.format(Xdomain.tolist(),Yvalues))
            plt.plot(Xdomain.tolist(), Yvalues,label='\u03C3='+str(sigma))
        except:
            print("unable to display {}".format(current_experiment))
    # Settings
    plt.title(model+" "+deformation+" certification")
    plt.xlabel('certification radius')
    plt.ylabel('certified accuracy')
    plt.ylim([0,1])
    plt.legend()
    plt.grid()
    counter+=1

plt.show()
