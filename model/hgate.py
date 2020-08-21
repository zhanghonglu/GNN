#异构图自编码器
import torch
from model.hgatConv import GATConv
from torch_geometric.nn import BatchNorm
class hgat_Encoder(torch.nn.Module):
    def __init__(self, node_featrues_num):
        super(hgat_Encoder, self).__init__()
        # 归一化
        self.norm = BatchNorm(in_channels=node_featrues_num, affine = False,track_running_stats=False)
        self.conv_near = GATConv(in_channels=node_featrues_num)
        self.conv_include = GATConv(in_channels=node_featrues_num)
        # self.conv_include = hgatConv(node_featrues_num=node_featrues_num)

    def forward(self, x, edge_index, edge_addr):
        x = self.norm(x)
        col,row = edge_index
        edge_index_include =torch.stack([col[edge_addr==0],row[edge_addr==0]],dim=0)
        x= self.conv_include(x, edge_index_include)
        edge_index_near = torch.stack([col[edge_addr == 1], row[edge_addr == 1]], dim=0)
        out = self.conv_near(x, edge_index_near)
        return  out

