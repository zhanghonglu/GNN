import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class GATConv(MessagePassing):

    def __init__(self, in_channels, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)
        #节点特征数量
        self.in_channels = in_channels
        #参数 衡量某种类型的边
        self.att = Parameter(torch.Tensor(in_channels, in_channels).double())

    def forward(self, x, edge_index, size=None):
        #聚合邻接向量
        return self.propagate(edge_index, size=size, x=x)

    def message(self,x, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        #目标节点的特征向量  [边的数目，节点的特征数目]
        x_i = x_j.view(-1, 1, self.in_channels)
        #源节点的特征向量    [边的数目，节点的特征数目]
        x_j = x_i.view(-1, self.in_channels, 1)
        # 计算注意力权重
        alpha = torch.matmul(torch.matmul(x_i,self.att),x_j)

        alpha = alpha.view(-1)
        #分组softmax
        alpha = softmax(alpha, edge_index_i, size_i)

        return x_j.view(-1,self.in_channels) * alpha.view(-1, 1)

    def update(self, aggr_out,x):
        #将聚合到的特征向量与原特征向量 平均
        return (x + aggr_out)/2
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                                             self.in_channels)
