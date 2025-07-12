import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetworkV5(nn.Module):
    def __init__(self, input_dim=34, hidden_dims=[128, 64], embedding_dim=256, dropout=0.2):
        super(SiameseNetworkV5, self).__init__()

        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final layer to embedding_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def embedding(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

    def forward(self, x1, x2):
        return self.embedding(x1), self.embedding(x2)
