import torch
from argparse import ArgumentParser
import os
from os import path
from DataLoaders import ModelNetLoader
import numpy as np
from Transformations.BatchReshapeForPointNet import reshapeBatchInput
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


dataset_choices = ['modelnet10', 'modelnet40']
model_choices = ['pointnet', 'pointnet++']
aug_choices = ['nominal', 'gaussianFull', 'rotation', 'translation', 'affine', 'scaling_uniform', 'DCT']

parser = ArgumentParser(description='PyTorch code for GeoCer')
parser.add_argument('--dataset', type=str, default='modelnet10', required=True, choices=dataset_choices)
parser.add_argument('--model', type=str, default='pointnet', required=True, choices=model_choices, help='model name for training')
parser.add_argument('--experiment_name', type=str, required=True, help='name of directory for saving results')
parser.add_argument('--aug_method', type=str, default='nominal', required=True, choices=aug_choices, help='type of augmentation for training')
parser.add_argument('--sigma', type=float, default=0.05, metavar='N', help='sigma value used for augmentation')
parser.add_argument('--batch_sz', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--seed', type=int, default=0, help='for deterministic behavior')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for training')
parser.add_argument('--step_sz', type=int, default=10, metavar='N', help='reducing the learning rate after this amount of epochs')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to saved checkpoint to load from')
parser.add_argument('--dataset_path', type=str, default='./Data/', help='name of directory contining the dataset')
parser.add_argument('--sampled_points', type=int, default=4096, help='points to be sampled from the mesh imported by torch geometric in case of modelnet')

args = parser.parse_args()


#
batch_exp_path = 'output/train/'
if not os.path.exists(batch_exp_path):
    os.makedirs(batch_exp_path, exist_ok=True)

# full path for output
args.output_path = os.path.join(batch_exp_path, args.experiment_name)
# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, exist_ok=True)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


# txt file with all params
args.info_log = os.path.join(args.output_path, f'info.txt')
print_training_params(args, args.info_log)

# final results
args.final_results = os.path.join(args.output_path, f'results.txt')



torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used: {}".format(device))

def train(epoch, model, train_loader, optimizer, writer, modelname,print_freq=100):
    model = model.train()
    
    MB = 1024.0**2
    GB = 1024.0**3
    for batch_idx,  (_, data, pointsPerCloud, target) in enumerate(train_loader):
        
        #these are tuples, the first position is a name, we are interested in the tensors, the second position
        data=data[1]
        pointsPerCloud = pointsPerCloud[1]
        target = target[1]


        #second position of the "ptr" array has the number of points sampled per cloud 
        pointsPerCloud = pointsPerCloud[1].item()

        #data needs to be of dimension Batches x pointsPerCloud x 3 for this implementation of pointnet to process it
        if modelname == 'pointnet':
            data = reshapeBatchInput(data,pointsPerCloud)
        
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        logits = model(data)
        
        loss = F.cross_entropy(logits, target)
        
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print(
                '+ Epoch: {}. Iter: [{}/{} ({:.0f}%)]. Loss: {:.5f}. '
                'Max mem: {:.2f}MB = {:.2f}GB.'
                .format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss / len(data),
                    torch.cuda.max_memory_allocated(device) / MB,
                    torch.cuda.max_memory_allocated(device) / GB,
                ),
                flush=True)

    writer.add_scalar('train/train_loss', loss, epoch)


def test(model, test_loader, device, writer, epoch,modelname):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, pointsPerCloud, target in test_loader:
            #these are tuples, the first position is a name, we are interested in the tensors, the second position
            data=data[1]
            pointsPerCloud = pointsPerCloud[1]
            target = target[1]


            #second position of the "ptr" array has the number of points sampled per cloud 
            pointsPerCloud = pointsPerCloud[1].item()

            #data needs to be of dimension Batches x pointsPerCloud x 3 for this implementation of pointnet to process it
            if modelname == 'pointnet':
                data = reshapeBatchInput(data,pointsPerCloud)
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            correct += (output.max(1)[1] == target).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('Test: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f})%)'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy)
        )

    writer.add_scalar('test/test_loss', test_loss, epoch)
    writer.add_scalar('test/test_accuracy', test_accuracy, epoch)

    return test_loss, test_accuracy


def contains_nan(model):
    for param in model.parameters():
        if torch.isnan(param.data).any():
            return True
    return False


def main(args):
    # load dataset
    if hasattr(ModelNetLoader, args.dataset):
        get_data_loaders = getattr(ModelNetLoader, args.dataset)
        if args.dataset == "modelnet10":
            num_classes = 10
        else:
            num_classes = 40
        
        train_loader = get_data_loaders("train",args.batch_sz,sampledPoints = args.sampled_points)
        test_loader = get_data_loaders("test",args.batch_sz,sampledPoints = args.sampled_points)
    else:
        raise Exception('Undefined Dataset')

    # load model
    if args.model == "pointnet":
        from Models.PointNet import PointNet
        base_classifier = PointNet(args.sampled_points,num_classes)
    else:
        raise Exception("Undefined model!")
    model = base_classifier
    if torch.cuda.is_available():
        model.cuda()
    #model = DeformWrapper(base_classifier, device, args.aug_method, args.sigma)
    # load optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.step_sz, gamma=0.1)

    epoch_init = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_param'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_init = checkpoint(['epoch'])
        print('Checkpoint is successfully loaded')



    for epoch in range(epoch_init, args.epochs):
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        train(epoch, model, train_loader, optimizer, args.writer,modelname = args.model, print_freq=args.print_freq)

        test(model, test_loader, device, writer, epoch,modelname = args.model)

        scheduler.step()

        # save model
        if not contains_nan(model):
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_param': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, f'{args.output_path}/FinalModel.pth.tar')
        else:
            break

    args.writer.close()


if __name__ == '__main__':

    # setup tensorboard
    tensorboard_path = f'tensorboard/{args.experiment_name}'
    if not path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path, flush_secs=10)
    args.writer = writer

    # main
    main(args)