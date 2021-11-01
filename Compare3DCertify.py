from __future__ import print_function
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import csv
from time import time
import datetime
import os
import math

certification_method_choices = ['rotationX','rotationY','rotationZ','rotationXZ','rotationXYZ','translation','shearing','tapering','twisting','squeezing','stretching','gaussianNoise','affine','affineNoTranslation'] 



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default='modelnet40', help="which dataset")
parser.add_argument("--model", type=str, default='pointnet', help="model name")
parser.add_argument('--num_points', type=int, default=64,help='number of points sampled from the meshes. make sure pretrained model matches this entry')
parser.add_argument('--max_features', type=int, default=1024,help='max features in Pointnet inner layers')
parser.add_argument("--base_classifier_path", type=str,default='Pointnet/trainedModels/AuthorsPretrained/64p_natural.pth', help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='rotationXYZ', required=True, choices=certification_method_choices, help='type of deformation for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True,help='name of directory for saving results')
parser.add_argument("--certify_batch_sz", type=int, default=256, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--chunks", type=int, default=1, help="how many chunks do we cut the test set into")
parser.add_argument("--num_chunk", type=int, default=0, help="which chunk to certify")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')
parser.add_argument('--cpuonly', action='store_true', default=False, help='force program to only use CPU')

args = parser.parse_args()

# full path for output
args.basedir = os.path.join('output/3DcertifyComparison', args.experiment_name)

# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.basedir):
    os.makedirs(args.basedir, exist_ok=True)
if not os.path.exists('output/samples/gaussianNoise'):
    os.makedirs('output/samples/gaussianNoise', exist_ok=True)
if not os.path.exists('output/samples/rotation'):
    os.makedirs('output/samples/rotation', exist_ok=True)
if not os.path.exists('output/samples/translation'):
    os.makedirs('output/samples/translation', exist_ok=True)
if not os.path.exists('output/samples/shearing'):
    os.makedirs('output/samples/shearing', exist_ok=True)
if not os.path.exists('output/samples/tapering'):
    os.makedirs('output/samples/tapering', exist_ok=True)
if not os.path.exists('output/samples/twisting'):
    os.makedirs('output/samples/twisting', exist_ok=True)
if not os.path.exists('output/samples/squeezing'):
    os.makedirs('output/samples/squeezing', exist_ok=True)
if not os.path.exists('output/samples/stretching'):
    os.makedirs('output/samples/stretching', exist_ok=True)
if not os.path.exists('output/samples/affineNoTranslation'):
    os.makedirs('output/samples/affineNoTranslation', exist_ok=True)
if not os.path.exists('output/samples/affine'):
    os.makedirs('output/samples/affine', exist_ok=True)

args.outfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.txt')


if __name__ == "__main__":


    if not args.cpuonly:
        #use cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # load model
    if args.model == 'pointnet':
        
        import sys
        sys.path.insert(0, "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet")

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from Pointnet.DataLoaders import datasets
        from torch.utils.data import DataLoader
        from Pointnet.model import PointNet
        from SmoothedClassifiers.CurveNetandPointnet.SmoothFlow import SmoothFlow

        if args.dataset == 'modelnet40':
            
            test_data = datasets.modelnet40(num_points=args.num_points, split='test', rotate='none')

            test_loader = DataLoader(
                dataset=test_data,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            num_classes = 40

            base_classifier = PointNet(
                number_points=args.num_points,
                num_classes=test_data.num_classes,
                max_features=args.max_features,
                pool_function='max',
                transposed_input= True
            )

            objective = nn.CrossEntropyLoss()
            optimizer = optim.Adam(base_classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

            #loadTrainedModel
            try:
                base_classifier.load_state_dict(torch.load(args.base_classifier_path,map_location=device))
            except:
                print('no pretrained model found')

            base_classifier = base_classifier.to(device)

            base_classifier.eval()

        elif args.dataset == 'modelnet10':
            raise NotImplementedError
        
        else:
            raise Exception("Undefined dataset!") 
        
    else:
        raise Exception("Undefined model!") 

       

    if args.certify_method == 'rotationZ' or args.certify_method == 'rotationXZ' or args.certify_method == 'rotationXYZ':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]
    # create the smooothed classifier g
    smoothed_classifier = SmoothFlow(base_classifier, num_classes, args.certify_method, args.sigma, device=device)

    # prepare output txt and csv files
    csvoutfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.csv')
    ftxt = open(args.outfile, 'w')
    fcsv = open(csvoutfile, 'w')

    # create the csv writer
    writer = csv.writer(fcsv)

    #print training params
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    print(text, file=ftxt)
    writer.writerow([str(key) + ': ' + str(d[key]) for key in d])

    #print header
    print("idx\t\tlabel\t\tpredict\t\tradius\t\tcorrect\t\ttime", file=ftxt, flush=True)
    writer.writerow(["idx","label","predict","radius","correct","time"])

    # iterate through the dataset
    dataset = [u for u in test_loader]

    interval = len(dataset)//args.chunks
    start_ind = args.num_chunk * interval

    #which pointcloud to take as sample in the output
    sampleNumber = 0

    #use same 100 examples as 3DCertify
    args.skip = len(dataset)//100
    
    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        #extract one at a time and the corresponding label
        x = dataset[i]
        if args.model == 'pointnet':
            label = x[2].item()
            x[1] = x[2]
            x[0] = x[0].to(device)
            x[1] = x[1].to(device)

        #only certify examples originally correctly classified by pointnet
        originalPrediction = base_classifier(x[0].permute(0,2,1)).argmax(1).item()
        if originalPrediction != label:
            continue
        
        #check if this is the pointcloud to sample
        if i == sampleNumber:
            plywrite = True
        else:
            plywrite = False

        before_time = time()
        # certify the prediction of g around x
        prediction, radius, p_A = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_batch_sz,plywrite)
        if args.uniform:
            radius = 2 * args.sigma * (p_A - 0.5)
        after_time = time()
        correct = int(prediction == label)
        print('Time spent certifying pointcloud {} was {} sec \t {}/{} ({:.2}%)'.format(i,after_time - before_time,i,start_ind + interval,100*i/(start_ind + interval)) )
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t\t{}\t\t{}\t\t{:.3}\t\t{}\t\t{}".format(i, label, prediction, radius, correct, time_elapsed), file=ftxt, flush=True)
        writer.writerow([i, label, prediction, radius, correct, time_elapsed])

    ftxt.close()
    fcsv.close()