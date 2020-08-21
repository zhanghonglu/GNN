import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE
from model.hgatConv import GATConv
from torch_geometric.utils import train_test_split_edges
torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'VGAE': VGAE}

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'Cora')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]
print(data)
pass

class Encoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        #聚合near类型的边的信息
        self.conv_near = GATConv(in_channels)
        #聚合include类型的边的信息
        self.conv_include = GATConv(in_channels)

    def forward(self, x, edge_index):
        #分别对不同类型的边进行处理
        x = self.conv_near(x, edge_index[:,:500])
        return self.conv_include(x, edge_index[:,500:])

channels = 16
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = kwargs[args.model](Encoder(dataset.num_features)).to(dev)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.double().to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    print(loss)
    loss.backward()
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     print(parameters.grad)
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 1000):
    train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))