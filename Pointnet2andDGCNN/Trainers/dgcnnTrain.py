#adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py
import os.path as osp
import os
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from DataLoaders import ScanobjectDataset

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

import sys
sys.path.insert(0, "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet2andDGCNN/Trainers/")
from pointnet2Train import MLP

class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(MLP([1024, 512]), Dropout(0.5), MLP([512, 256]),
                       Dropout(0.5), Lin(256, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


def train(epoch):
    model.train()
    counter = 0
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        if counter % 10 == 0:
            print(
                '+ Epoch: {}. Iter: [{}/{} ({:.0f}%)]. Loss: {:.5f}. '
                .format(
                    epoch,
                    counter * len(data.y),
                    len(train_loader.dataset),
                    100. * counter / len(train_loader),
                    loss / len(data),
                ),
                flush=True)
        counter += 1
    writer.add_scalar('train/train_loss', loss, epoch)
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
            test_loss += F.nll_loss(model(data), data.y).item()
        correct += pred.eq(data.y).sum().item()
    
    test_loss /= len(loader.dataset)
    test_accuracy = 100. * correct / len(loader.dataset)
    print('Test: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f})%)'.format(
        test_loss, correct, len(loader.dataset), test_accuracy)
        )

    writer.add_scalar('test/test_loss', test_loss, epoch)
    writer.add_scalar('test/test_accuracy', test_accuracy, epoch)
    return correct / len(loader.dataset)

#utils for log
def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)

def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


if __name__ == '__main__':

    dataset_choices = ['modelnet40','modelnet10','scanobjectnn']
    
    #arguments passed
    parser = ArgumentParser(description='PyTorch code for GeoCer')
    parser.add_argument("--dataset", default='modelnet40',choices=dataset_choices, help="which dataset")
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--num_points", type=int, default=1024,help="amount of points to sample per pointcloud")
    parser.add_argument('--experiment_name', type=str, default='tutorial', required=True)
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    args = parser.parse_args()

    # setup tensorboard
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_path = f'../tensorboard/{args.experiment_name}'
    if not osp.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path, flush_secs=10)

    #dirs to output results
    batch_exp_path = '../trainedModels/'
    if not os.path.exists(batch_exp_path):
        os.makedirs(batch_exp_path, exist_ok=True)

    # full path for output
    output_path = os.path.join(batch_exp_path, args.experiment_name)

    # Log path: verify existence of output_path dir, or create it
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # txt file with all params
    info_log = os.path.join(output_path, f'info.txt')
    print_training_params(args, info_log)

    
    if args.dataset == 'modelnet40':
        #dataset and loaders
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                        'Data/Modelnet40fp')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
        train_dataset = ModelNet(path, '40', True, transform, pre_transform)
        test_dataset = ModelNet(path, '40', False, transform, pre_transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                                num_workers=6)
        num_classes = train_dataset.num_classes

    elif args.dataset == 'modelnet10':
        #dataset and loaders
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                        'Data/Modelnet10fp')
        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
        train_dataset = ModelNet(path, '10', True, transform, pre_transform)
        test_dataset = ModelNet(path, '10', False, transform, pre_transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                                num_workers=6)
        num_classes = train_dataset.num_classes
    
    elif args.dataset == 'scanobjectnn':
        train_dataset = ScanobjectDataset.ScanObjectNN(args.data_dir, 'train',  args.num_points,
                                variant='obj_only', dset_norm="inf")
        test_dataset = ScanobjectDataset.ScanObjectNN(args.data_dir, 'test',  args.num_points,
                                variant='obj_only', dset_norm="inf")
        classes = train_dataset.classes
        num_classes = len(classes)

        train_loader = DataLoader(train_dataset, batch_size=32,
                                shuffle=True, num_workers=6, drop_last=True)

        test_loader = DataLoader(train_dataset, batch_size=32,
                                shuffle=False, num_workers=6)

    #model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_classes, k=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    try:
        #loadTrainedModel
        checkpoint = torch.load('../trainedModels/' + args.experiment_name + '/FinalModel.pth.tar')
        model.load_state_dict(checkpoint['model_param'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        print('no pretrained model found')


    bestTestAcc = -1 
    #train and test
    for epoch in range(1, args.epochs):
        loss = train(epoch)
        test_acc = test(test_loader)
        print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, test_acc))
        scheduler.step()
        if test_acc >= bestTestAcc:
            torch.save(
                        {
                            'epoch': epoch + 1,
                            'model_param': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        }, f'{output_path}/FinalModel.pth.tar')
            bestTestAcc = test_acc
        