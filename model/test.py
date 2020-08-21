import torch
import torch.nn.functional as F
from covid_data.subGraph import get_subGraph
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE,GATConv
from model.utils.split_edge import train_test_split_edges
from model.hgate import hgat_Encoder
from torch_geometric.data import Data

x = torch.rand(50,5).double()
edge_index = torch.randint(

).long()
edge_addr = torch.Tensor([[1], [0], [1]])
torch.manual_seed(12345)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data  = Data(x=x, edge_index =edge_index, edge_attr = edge_addr)
model = GAE(hgat_Encoder(node_featrues_num=5)).double().to(dev)
x, train_pos_edge_index,train_pos_edge_attr = data.x.to(dev), data.edge_index.to(dev), data.edge_attr.to(dev)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.01)






if __name__ == '__main__':


    for i in range(1,1000):
        model.train()
        # optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        print(z)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(x,z)
        print("当前loss{}".format(loss))
        loss.backward()
        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.size())
        #     print(parameters.grad)
        optimizer.step()