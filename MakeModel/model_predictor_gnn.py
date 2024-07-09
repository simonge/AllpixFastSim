import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

######################################################
# Define the position, momentum and time graph neural network predictormodel
######################################################
class Predictor(nn.Module):
    def __init__(self, nNodeParameters=4, nPredictions=5):
        super(Predictor, self).__init__()
        self.nNodeParameters = nNodeParameters
        self.nPredictions = nPredictions
        self.conv1 = GCNConv(nNodeParameters, 16)
        self.conv2 = GCNConv(16, 8)
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, nPredictions)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        

######################################################