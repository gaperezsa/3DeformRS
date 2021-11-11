#adapted from https://github.com/eth-sri/3dcertify/blob/master/train_classification.py
import argparse
import os
import datetime
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DataLoaders import datasets
from model import PointNet
datasetChoices = ['modelnet40','scanobjectnn']

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True, help='willbe used as the name of the output dir')
parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use', choices=datasetChoices)
parser.add_argument('--data_dir', type=str, default='Data/', help='path to the raw data')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=1024, help='number of points per point cloud')
parser.add_argument('--seed', type=int, default=123456, help='seed for random number generator')
parser.add_argument('--ignore_existing_output_dir', action='store_false', help='ignore if output dir exists')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel data loader workers')
#parser.add_argument('--defense', action='store_true', help='use adversarial training')
#parser.add_argument('--eps', type=float, default=0.02, help='radius of eps-box to defend around point')
#parser.add_argument('--step_size', type=float, default=None, help='step size for FGSM')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--max_features', type=int, default=1024, help='the number of features for max pooling')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='global pooling function')
#parser.add_argument('--domain', choices=['box', 'face'], default='box', help='The attack model domain')
parser.add_argument('--rotation', choices=['none', 'z', 'so3'], default='z', help='Axis for rotation augmentation in modelnet40')

settings = parser.parse_args()

settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
settings.out = os.path.join('trainedModels', settings.experiment_name)

#defense code not used
#if not settings.step_size:
#   settings.step_size = 1.25 * settings.eps

os.makedirs(settings.out, exist_ok=settings.ignore_existing_output_dir)

#create logger
def create_logger(log_dir: Union[str, Path], log_name: str = "certification") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(Path(log_dir) / f"{log_name}_{datetime.datetime.now().isoformat(timespec='seconds')}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
#log_name = f"train_defended[{settings.defense}]_eps[{settings.eps}]_rotation[settings.rotation]_pooling[{settings.pooling}]"
log_name = f"rotation[settings.rotation]_pooling[{settings.pooling}]"

logger = create_logger(settings.out, log_name)

logger.info(settings)

writer = SummaryWriter(log_dir=os.path.join('tensorboard/', settings.experiment_name))



if settings.dataset == 'modelnet40':
    train_data = datasets.modelnet40(num_points=settings.num_points, split='train', rotate=settings.rotation)
    test_data = datasets.modelnet40(num_points=settings.num_points, split='test', rotate='none')

    num_classes = train_data.num_classes

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers
    )

    print("Train Size: ", len(train_data))
    print("Test Size: ", len(test_data))
    print("Total Size: ", len(test_data) + len(train_data))

    distribution = np.zeros(40, dtype=int)
    for sample in train_data:
        _, _, label = sample
        distribution[label.item()] += 1
    print(distribution)

elif settings.dataset == 'scanobjectnn':
    train_data = datasets.ScanObjectNN(settings.data_dir, 'train',  settings.num_points,
                              variant='obj_only', dset_norm="inf")
    test_data = datasets.ScanObjectNN(settings.data_dir, 'test',  settings.num_points,
                            variant='obj_only', dset_norm="inf")
    classes = train_data.classes
    num_classes = len(classes)

    train_loader = DataLoader(train_data, batch_size=settings.batch_size,
                            shuffle=True, num_workers=settings.num_workers,collate_fn=datasets.collate_fn, drop_last=True)

    test_loader = DataLoader(test_data, batch_size=settings.batch_size,
                            shuffle=False, num_workers=settings.num_workers,collate_fn=datasets.collate_fn)

num_batches = len(train_data) / settings.batch_size
logger.info("Number of batches: %d", num_batches)
logger.info("Number of classes: %d", num_classes)
logger.info("Training set size: %d", len(train_data))
logger.info("Test set size: %d", len(test_data))

model = PointNet(
    number_points=settings.num_points,
    num_classes=num_classes,
    max_features=settings.max_features,
    pool_function=settings.pooling
)
model = model.to(settings.device)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

logger.info("starting training")

bestTestAcc = -1 
for epoch in range(settings.epochs):

    train_correct = 0
    train_amount = 0
    train_loss = 0

    for i, data in enumerate(tqdm(train_loader)):
        points, faces, label = data
        label: torch.Tensor = torch.squeeze(label)
        points: torch.Tensor = points.float().to(settings.device)
        #faces: torch.Tensor = faces.float().to(settings.device)
        label: torch.Tensor = label.to(settings.device)
        
        #defense code not used
        '''
        if settings.defense:
            if settings.domain == "box":
                domain = attacks.EpsBox(points, settings.eps)
            elif settings.domain == "face":
                domain = attacks.FaceBox(faces)
            else:
                assert False, f"Unsupported domain {settings.domain}"

            model.eval()
            points = domain.random_point()
            points = attacks.fgsm(model, points, label, step_size=settings.step_size)
            points = domain.project(points)
        '''

        model.train()
        optimizer.zero_grad()

        predictions = model(points)
        loss = objective(predictions, label)
        loss.backward()
        optimizer.step()

        max_predictions = predictions.data.max(1)[1]
        correct = max_predictions.eq(label.data).cpu().sum()
        train_correct += correct.item()
        train_amount += points.size()[0]
        train_loss += loss.item()

    test_correct = 0
    test_amount = 0
    test_loss = 0

    for i, data in enumerate(test_loader):
        points, _, label = data
        points = points[:, : settings.num_points, :]
        label = torch.squeeze(label)
        points = points.to(settings.device)
        label = label.to(settings.device)

        model = model.eval()
        predictions = model(points)
        loss = objective(predictions, label)
        max_predictions = predictions.data.max(1)[1]
        correct = max_predictions.eq(label.data).cpu().sum()
        test_correct += correct.item()
        test_amount += points.size()[0]
        test_loss += loss.item()
    
    test_accuracy=test_correct / test_amount
    logger.info(
        "Epoch {epoch}: train loss: {train_loss}, train accuracy: {train_accuracy}, test loss: {test_loss}, test accuracy: {test_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_correct / train_amount,
            test_loss=test_loss,
            test_accuracy=test_correct / test_amount
        )
    )

    if test_accuracy >= bestTestAcc:
            torch.save(
                        {
                            'epoch': epoch + 1,
                            'model_param': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        }, f'{settings.out}/FinalModel.pth.tar')
            bestTestAcc = test_accuracy

    writer.add_scalar('accuracy/train', train_correct / train_amount, epoch)
    writer.add_scalar('loss/train', train_loss / train_amount, epoch)
    writer.add_scalar('accuracy/test', test_correct / test_amount, epoch)
    writer.add_scalar('loss/test', test_loss / test_amount, epoch)

    scheduler.step()
