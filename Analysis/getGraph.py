import pandas as pd
import numpy as np
import matplotlib
import argparse
import math
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import glob
from sklearn import metrics
sns.set_theme()
sns.set_style("darkgrid")


'''
This program will be able to retrieve the graph for a list of models, running some kind of deformation and given the sigmas interested in.
This is all asuming that when experiments where run, the convention {model}{deformation}{dataset spec if needed}{sigma} was followed
examples: pointnet2GaussianNoise0.02 , dgcnnShearing0.4 , dgcnnAffineModelnet10_0.005
'''
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument('--parallel', action='store_true', default=False, help='add flag to use parallel computation of the graphs')
parser.add_argument('--envelope', action='store_true', default=False, help='add flag to make the envelope curve appear solid and the other as dashlines underneath it')
parser.add_argument('--hypervolume', action='store_true', default=False, help='add flag to make the X axis in terms of hypervolume certified')
parser.add_argument('--less_labels', action='store_true', default=False, help='add flag to legend just half of the curves')
args = parser.parse_args()

#change these as needed for current query
models=["Pointnet2"]#,"pointnet2","dgcnn","curvenet"]
deformation="RotationZ"
#sigmas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]#[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3]
counter = 0
base_path = "../output/certify/scanobjectnn/RotationZ/"
save_path = '/home/santamgp/Downloads/CVPRGraphics/TestingREADME/'
current_experiment = ""


#function used if parallel computation asked
def checkDfs(domainValue):
    return max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/(df.count()[0]) for df in dfs])

#function used if parallel computation asked
def allYvals(dfsAndXdomain):
    df,Xdomain = dfsAndXdomain
    return [( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/(df.count()[0]) for i in Xdomain]

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

if(len(models) == 1):
    axes=[1]
    fig,axes[0] = plt.subplots(1,len(models),sharey=True)#figsize=(10, 4)
    fig.suptitle(str(deformation),fontsize=20)
else:
    fig,axes = plt.subplots(1,len(models),sharey=True)#figsize=(10, 4)
    fig.suptitle(str(deformation),fontsize=20)
for model in models:
    try:
        csvPaths = sorted(glob.glob(f"{base_path}/*{model}{deformation}*/*.csv"))
        sigmasBuilder = sorted(glob.glob(f"{base_path}/*{model}{deformation}[0-9]*"))
        sigmas = [ i.split(deformation)[-1] for i in sigmasBuilder]
        sigmas = [float(s) for s in sigmas]
        dfs = [pd.read_csv(csvPath,skiprows=1) for csvPath in csvPaths] #first row is the command used, not needed here
        totalRows = max([df.count()[0] for df in dfs])
        Xdomains = [np.append(np.linspace(0,df["radius"].max(),num=totalRows) ,df["radius"].max() + (df["radius"].max()/(totalRows-1)) ) for df in dfs]
        if args.parallel:
            pool_obj = multiprocessing.Pool()
            Yvalues = pool_obj.map(allYvals,zip(dfs,Xdomains))
        else:
            Yvalues = [[( (df["correct"] == 1) & (df["radius"] >= i) ).sum()/(df.count()[0]) for i in Xdomain] for df,Xdomain in zip(dfs,Xdomains)]
        

        #useful when less_labels
        showLegendCounter = 0

        for Xdomain,sigma,Yvalue in zip(Xdomains,sigmas,Yvalues):
            if (args.hypervolume):
                plottingDomain = DomainTransformer(deformation,Xdomain)
            else:
                plottingDomain = Xdomain

            if (args.less_labels):
                label = '\u03C3='+str(sigma) if showLegendCounter%2==1 else '_\u03C3='+str(sigma) #those labels starting with _ are ignored by the automatic legend
                showLegendCounter += 1
            else :
                label = '\u03C3='+str(sigma)

            sns.lineplot(ax=axes[counter],x=plottingDomain.tolist(),y=Yvalue,label=label)

        if args.envelope:
            print('calculating envelope...')
            maxRadius = max([ Xdomain[-1] for Xdomain in Xdomains ])
            EnvelopeXdomain = np.append(np.linspace(0,maxRadius,num=totalRows*len(sigmas)) ,maxRadius + maxRadius/(totalRows-1))

            if args.parallel:
                second_pool_obj = multiprocessing.Pool()
                EnvelopeYvalues = second_pool_obj.map(checkDfs,EnvelopeXdomain)
            else:
                EnvelopeYvalues = [max([( (df["correct"] == 1) & (df["radius"] >= domainValue) ).sum()/(df.count()[0]) for df in dfs]) for domainValue in EnvelopeXdomain]
            
            if (args.hypervolume):
                plottingDomain = DomainTransformer(deformation,EnvelopeXdomain)
            else:
                plottingDomain = EnvelopeXdomain

            sns.lineplot(ax=axes[counter],x=plottingDomain.tolist(), y=EnvelopeYvalues,label=f'Ours (Envelope) \nACR={metrics.auc(plottingDomain.tolist(), EnvelopeYvalues):.2f}')
            #plt.plot(plottingDomain.tolist(), EnvelopeYvalues,label='envelope')
            print('done!\n')
        
        #draw other paper's points
        if (model == "64pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,88/90,'ro')#3 degreees
            axes[counter].plot(0.349066,0.967,'ro',label='3D certify')#20 degreees, 96.7%
            axes[counter].plot(1.0472,0.957,'ro')#60 degrees, 95.7%
        elif (model == "64pointnet" and deformation == "RotationX" and args.envelope):
            axes[counter].plot(0.0174533,88/90,'ro',label='3D certify')#1 degree
            axes[counter].plot(0.0349066,86/90,'ro') #2 degree
            axes[counter].plot(0.0523599,86/90,'ro') #3 degree, 
            axes[counter].plot(0.0698132,86/90,'ro') #4 degree, 
            axes[counter].plot(0.0872665,84/90,'ro') #5 degree, 
            axes[counter].plot(0.10472,84/90,'ro')   #6 degree, 
            axes[counter].plot(0.122173,79/90,'ro')  #7 degree, 
            axes[counter].plot(0.139626,76/90,'ro')  #8 degree, 
            axes[counter].plot(0.174533,71/90,'ro')  #10 degree, 
            #plt.plot(0.261799,50/90,'ro')  #15 degree, 
        elif (model == "64pointnet" and deformation == "RotationY" and args.envelope):
            axes[counter].plot(0.0174533,89/90,'ro',label='3D certify')#1 degree, 98.8%
            axes[counter].plot(0.0349066,89/90,'ro') #2 degree,
            axes[counter].plot(0.0523599,89/90,'ro') #3 degree, 
            axes[counter].plot(0.0698132,88/90,'ro') #4 degree, 
            axes[counter].plot(0.0872665,86/90,'ro') #5 degree, 
            axes[counter].plot(0.10472,86/90,'ro')   #6 degree, 
            axes[counter].plot(0.122173,85/90,'ro')  #7 degree, 
            axes[counter].plot(0.139626,81/90,'ro')  #8 degree, 
            axes[counter].plot(0.174533,76/90,'ro')  #10 degree, 
            #plt.plot(0.261799,50/90,'ro')  #15 degree,
        elif (model == "32pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,0.94,'ro',label='3D certify')#3 degreees
        elif (model == "128pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,0.811,'ro',label='3D certify')#3 degreees
        elif (model == "256pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,0.667,'ro',label='3D certify')#3 degreees
        elif (model == "512pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,0.494,'ro',label='3D certify')#3 degreees
        elif (model == "1024pointnet" and deformation == "RotationZ" and args.envelope):
            axes[counter].plot(0.0523599,0.371,'ro',label='3D certify')#3 degreees
    except:
        print("unable to display {}".format(current_experiment))
    

    # Settings, change these to your liking
    axes[counter].set_title(f'{model}')

    if (args.hypervolume):
        axes[counter].set_xlabel('certified hypervolume')
    elif deformation[:8]=="Rotation":
        axes[counter].set_xlabel('Radians',fontsize=20)
    else:
        axes[counter].set_xlabel('certification radius',fontsize=20)

    if args.envelope:
        for line in axes[counter].lines[0:len(sigmas)]:
            line.set_linestyle("--")

    axes[counter].set_ylabel('Certified Accuracy',fontsize=20)
    axes[counter].set_ylim([0,1])
    axis=axes[counter].tick_params(labelsize=14)
    # try:
    #     axes[counter].get_legend().remove()
    # except:
    #     print("warning removing labels")
    #plt.grid()
    axes[counter].legend(loc='upper right', bbox_to_anchor=(1, 1),framealpha=0.5)
    counter+=1


plt.savefig(f"{save_path}{deformation}.png",bbox_inches='tight')
plt.savefig(f"{save_path}{deformation}.pdf",bbox_inches='tight')
plt.savefig(f"{save_path}{deformation}.eps",bbox_inches='tight')
plt.show()
