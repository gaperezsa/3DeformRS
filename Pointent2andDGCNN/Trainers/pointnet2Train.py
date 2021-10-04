#adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py

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


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self,out_channels):
        super(Net, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, out_channels)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()
    counter = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
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


def test(loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data in loader:
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

    #arguments passed
    parser = ArgumentParser(description='PyTorch code for GeoCer')
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
    batch_exp_path = '../output/train/'
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




    #dataset and loaders
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'Data/Modelnet40fp')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '40', True, transform, pre_transform)
    test_dataset = ModelNet(path, '40', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    #model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #train and test
    for epoch in range(1, args.epochs):
        train(epoch)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
    torch.save(
                {
                    'epoch': epoch + 1,
                    'model_param': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'{output_path}/FinalModel.pth.tar')