import torch
import torch.nn.functional as F
from covid_data.subGraph import get_subGraph
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE,GATConv
from model.utils.split_edge import train_test_split_edges
from model.hgate import hgat_Encoder

data = get_subGraph(1, "New York")
torch.manual_seed(12345)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
model = GAE(hgat_Encoder(node_featrues_num=5)).to(dev).double()
x, train_pos_edge_index, train_pos_edge_addr= data.x.double().to(dev), data.train_pos_edge_index.to(dev), data.train_pos_edge_attr.to(dev)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.000005)


def train():
    model.train()
    optimizer.zero_grad()
    # for name, parameters in model.named_parameters():
    #     print(parameters.grad)
    z = model.encode(x, train_pos_edge_index,train_pos_edge_addr)
    loss = model.recon_loss(z, train_pos_edge_index)
    print(loss)
    loss.backward()
    # for name, parameters in model.named_parameters():
    #     print(parameters.grad)
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index, train_pos_edge_addr)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 1000000):
    train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
