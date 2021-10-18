import torch
import torch.nn as nn

class PointNet(nn.Module):

    def __init__(
            self,
            number_points: int,
            num_classes: int,
            max_features: int = 1024,
            pool_function: str = 'max',
            disable_assertions: bool = False,
            transposed_input: bool = False
    ):
        super(PointNet, self).__init__()
        assert number_points % 8 == 0, f"Number of points must be divisible by 8: {number_points}"
        self.number_points = number_points
        self.max_features = max_features
        self.disable_assertions = disable_assertions
        self.transposed_input = transposed_input

        # input-dimension: (batch_size, features (coordinates), number_points)
        # First MLP with weight sharing, implemented as 1d convolution
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )

        # Second MLP with weight sharing, implemented as 1d convolution
        # dimension: (batch_size, 64, number_points)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=self.max_features, kernel_size=1),
            nn.BatchNorm1d(num_features=self.max_features),
            nn.ReLU()
        )
        # dimension: (batch_size, 1024, number_points)

        # global pooling of features across points
        pooling = []
        if pool_function == 'improved_max':
            remaining_dim = self.number_points
            while remaining_dim > 8:
                assert remaining_dim % 8 == 0, "number_points must be recursively divisible by 8"
                pooling.append(nn.MaxPool1d(kernel_size=8, stride=8))
                remaining_dim = remaining_dim // 8
            pooling.append(nn.MaxPool1d(kernel_size=remaining_dim, stride=remaining_dim))
        elif pool_function == 'max':
            pooling.append(nn.MaxPool1d(kernel_size=number_points))
        elif pool_function == 'avg':
            pooling.append(nn.AvgPool1d(kernel_size=number_points))
        else:
            assert False, f"Invalid pooling operation {pool_function}"
        self.pooling = nn.Sequential(*pooling)
        print(self.pooling)
        # dimension: (batch_size, 1024, 1)

        # global fully connected layers
        # dimension: (batch_size, 1024)
        self.mlp3 = nn.Sequential(
            nn.Linear(in_features=self.max_features, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        # dimension: (batch_size, num_classes)

    def forward(self, x):
        if not self.disable_assertions and not self.transposed_input:
            assert x.dim() == 3, f"Expect input with 3 dimensions: (batch x num_points x features), got {x.size()}."
            assert x.size(1) == self.number_points, f"Expect input of size (N x num_points x features), got {x.size()}."
            assert x.size(2) == 3, f"Expect input of size (N x num_points x features), got {x.size()}."
        # input is in (batch x num_points x features), but we need (batch x features x num_points) for conv layers
        if not self.transposed_input:
            x = torch.transpose(x, 2, 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.pooling(x)
        x = torch.reshape(x, shape=(-1, self.max_features))
        x = self.mlp3(x)
        return x
