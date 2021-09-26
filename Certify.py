import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from deform_smooth import SmoothFlow
from time import time
import datetime
import os
import math
from torchvision.models.resnet import resnet50

dataset_choices = ['modelnet40']
model_choices = ['pointnet2','dgcnn']
certification_method_choices = ['rotation'] #'nominal', 'gaussianFull', 'rotation', 'translation', 'affine', 'scaling_uniform' ,'DCT'



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default='modelnet40',choices=dataset_choices, help="which dataset")
parser.add_argument("--model", type=str, choices=model_choices, help="model name")
parser.add_argument("--base_classifier_path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='rotation', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True,help='name of directory for saving results')
parser.add_argument("--certify_batch_sz", type=int, default=32, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--chunks", type=int, default=1, help="how many chunks do we cut the test set into")
parser.add_argument("--num_chunk", type=int, default=0, help="which chunk to certify")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')

args = parser.parse_args()

# full path for output
args.basedir = os.path.join('output/certify', args.experiment_name)
# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.basedir):
    os.makedirs(args.basedir, exist_ok=True)
args.outfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.txt')

def copy_pretrained_model(model, path_to_copy_from):
    checkpoint = torch.load(path_to_copy_from)['model_param']
    # print(resnet.keys())
    keys = list(checkpoint.keys())
    count = 0
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(checkpoint[keys[count]].data)
        count +=1
    model = model.to('cuda')
    print('Pretrained model is loaded successfully')
    return model

if __name__ == "__main__":

    # load model
    if args.model == 'pointnet2':
        from Pointent2andDGCNN.pointnet2Train import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader

        if args.dataset == 'modelnet40':
            
            #dataset and loaders
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)


            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            #loadTrainedModel
            checkpoint = torch.load('../output/train/' + args.experiment_name + '/FinalModel.pth.tar')
            model.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.model == 'dgcnn':
        from Pointent2andDGCNN.dgcnnTrain import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader

        if args.dataset == 'modelnet40':

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Pointent2andDGCNN/Data/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                        num_workers=6)

            num_classes = 40
            #model and optimizer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(num_classes, k=20).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            #loadTrainedModel
            checkpoint = torch.load('../output/train/' + args.experiment_name + '/FinalModel.pth.tar')
            model.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        raise Exception("Undefined model!") 

       

    if args.certify_method == 'rotation':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]
    # create the smooothed classifier g
    smoothed_classifier = SmoothFlow(base_classifier, num_classes, args.certify_method, args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = test_loader.dataset

    interval = len(dataset)//args.chunks
    start_ind = args.num_chunk * interval

    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius, p_A = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_batch_sz)
        if args.uniform:
            radius = 2 * args.sigma * (p_A - 0.5)
        after_time = time()
        correct = int(prediction == label)
        print('Time for certifying one image is', after_time - before_time )
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()