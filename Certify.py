import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import csv
from time import time
import datetime
import os
import math
from torchvision.models.resnet import resnet50

dataset_choices = ['modelnet40']
model_choices = ['pointnet2','dgcnn']
certification_method_choices = ['rotation','translation','shearing','tapering','twisting','squeezing','gaussianNoise','affine'] 



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default='modelnet40',choices=dataset_choices, help="which dataset")
parser.add_argument("--model", type=str, choices=model_choices, help="model name")
parser.add_argument("--base_classifier_path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='rotation', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True,help='name of directory for saving results')
parser.add_argument("--certify_batch_sz", type=int, default=150, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--chunks", type=int, default=1, help="how many chunks do we cut the test set into")
parser.add_argument("--num_chunk", type=int, default=0, help="which chunk to certify")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')

args = parser.parse_args()

#squeezing only defined for -1 < sigma < 1
if args.certify_method == 'squeezing' and (args.sigma > 1 or args.sigma < -1 ):
    print("certifying for squeezing with sigma : {} is not defined here, setting sigma to 0.999999".format(args.sigma))
    args.sigma = 0.999999

# full path for output
args.basedir = os.path.join('output/certify', args.experiment_name)

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
if not os.path.exists('output/samples/affine'):
    os.makedirs('output/samples/affine', exist_ok=True)

args.outfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.txt')


if __name__ == "__main__":

    # load model
    if args.model == 'pointnet2':
        from Pointent2andDGCNN.Trainers.pointnet2Train import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader
        from SmoothedClassifiers.Pointnet2andDGCNN.SmoothFlow import SmoothFlow

        if args.dataset == 'modelnet40':
            
            #dataset and loaders
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_classifier = Net(num_classes).to(device)
            optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)

            #loadTrainedModel
            checkpoint = torch.load(args.base_classifier_path)
            base_classifier.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        elif args.dataset == 'modelnet10':
            
            #dataset and loaders
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet10fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
            print(path)
            test_dataset = ModelNet(path, '10', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_classifier = Net(num_classes).to(device)
            optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)

            #loadTrainedModel
            checkpoint = torch.load(args.base_classifier_path)
            base_classifier.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.model == 'dgcnn':
        from Pointent2andDGCNN.Trainers.dgcnnTrain import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader
        from SmoothedClassifiers.Pointnet2andDGCNN.SmoothFlow import SmoothFlow

        if args.dataset == 'modelnet40':

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)

            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_classifier = Net(num_classes, k=20).to(device)
            optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            #loadTrainedModel
            checkpoint = torch.load(args.base_classifier_path)
            base_classifier.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

        elif args.dataset == 'modelnet10':

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet10fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
            print(path)
            test_dataset = ModelNet(path, '10', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)

            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            base_classifier = Net(num_classes, k=20).to(device)
            optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            #loadTrainedModel
            checkpoint = torch.load(args.base_classifier_path)
            base_classifier.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        raise Exception("Undefined model!") 

       

    if args.certify_method == 'rotation':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]
    # create the smooothed classifier g
    smoothed_classifier = SmoothFlow(base_classifier, num_classes, args.certify_method, args.sigma)

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
    
    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        
        #check if this is the pointcloud to sample
        if i == sampleNumber:
            plywrite = True
        else:
            plywrite = False


        #extract one at a time
        x = dataset[i]
        label = x.y.item()

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
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