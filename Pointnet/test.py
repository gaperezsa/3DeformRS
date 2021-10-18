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

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True, help='path of output directory')
parser.add_argument('--dataset', type=str, default='modelnet40', help='the dataset to use', choices=['modelnet40'])
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--num_points', type=int, default=1024, help='number of points per point cloud')
parser.add_argument('--num_workers', type=int, default=0, help='number of parallel data loader workers')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--max_features', type=int, default=1024, help='the number of features for max pooling')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='global pooling function')

settings = parser.parse_args()

settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_data = datasets.modelnet40(num_points=settings.num_points, split='test', rotate='none')

test_loader = DataLoader(
    dataset=test_data,
    batch_size=settings.batch_size,
    shuffle=False,
    num_workers=settings.num_workers
)

print("Test Size: ", len(test_data))
distribution = np.zeros(40, dtype=int)
for sample in test_data:
    _, _, label = sample
    distribution[label.item()] += 1
print(distribution)

model = PointNet(
    number_points=settings.num_points,
    num_classes=test_data.num_classes,
    max_features=settings.max_features,
    pool_function=settings.pooling
)
model = model.to(settings.device)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

#loadTrainedModel
try:
    checkpoint = torch.load('trainedModels/' + settings.experiment_name + '/FinalModel.pth.tar')
    model.load_state_dict(checkpoint['model_param'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
except:
    #before saying there is no model check if it is the 3d certify authors pretrained model
    try:
        model.load_state_dict(torch.load('trainedModels/' + settings.experiment_name + '/1024p_natural.pth'))
    except:
        print('no pretrained model found')


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

print("test loss: {test_loss}, test accuracy: {test_accuracy}".format(
        test_loss=test_loss,
        test_accuracy=test_correct / test_amount
        )
    )