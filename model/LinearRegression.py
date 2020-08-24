import  torch
import torch.nn.functional as F

class LinearRegression(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output
    features are defaults both 1
    """

    def __init__(self, feature_num):
        super().__init__()
        self.linear = torch.nn.Linear(feature_num, 1)
        self.bias = torch.nn.Parameter(torch.Tensor(feature_num))

    def forward(self, x):
        out = self.linear(x)+self.bias
        return F.relu(out)