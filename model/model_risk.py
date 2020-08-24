import torch
from model.hgate import hgat_Encoder
from torch_geometric.nn import GAE
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(hgat_Encoder(node_featrues_num=5)).to(dev).double()
