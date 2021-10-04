
import os
import os.path as osp
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import pointnet2Train



#arguments passed
parser = ArgumentParser(description='PyTorch code for GeoCer')
parser.add_argument('--experiment_name', type=str, default='tutorial', required=True)
args = parser.parse_args()


#dataset and loaders
path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                'Data/Modelnet40fp')
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
print(path)
test_dataset = ModelNet(path, '40', False, transform, pre_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=6)

#model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pointnet2Train.Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#loadTrainedModel
checkpoint = torch.load('../output/train/' + args.experiment_name + '/FinalModel.pth.tar')
model.load_state_dict(checkpoint['model_param'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
test_loss = 0
correct = 0
for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        pred = model(data).max(1)[1]
        test_loss += F.nll_loss(model(data), data.y).item()
    correct += pred.eq(data.y).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = 100. * correct / len(test_loader.dataset)
print('Test: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f})%)'.format(
    test_loss, correct, len(test_loader.dataset), test_accuracy)
    )