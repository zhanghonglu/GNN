import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data


class hgatConv(MessagePassing):

    def __init__(self, node_featrues_num ):
        super().__init__(aggr='add')
        self.node_featrues_num = node_featrues_num
        self.att= Parameter(torch.Tensor(node_featrues_num, node_featrues_num)).double()
        # self.att = torch.nn.Linear(in_features=node_featrues_num,out_features=node_featrues_num)


    def forward(self,x, edge_index):
        i,j = edge_index
        x_i_attr = x[i]
        x_j_attr = x[j]
        weight_0 = torch.matmul(x_i_attr,self.att)
        weight_1 = torch.bmm(weight_0.unsqueeze(1), x_j_attr.unsqueeze(2))
        weight_2 = weight_1.squeeze()
        weight_soft = softmax(weight_2,i)

        return self.propagate(edge_index, x=x, weight_soft = weight_soft)

    def message(self,x_j,weight_soft):
        #传递每一个边的信息，从源节点j 到目标节点i
        message = x_j*weight_soft.view(-1,1)

        return x_j*weight_soft.view(-1,1)




if __name__ == '__main__':
    x = torch.Tensor([[3,1],[4,1],[5,3],[4,2]]).float()
    print(x)
    edge_index = torch.Tensor([[0,3,0],[1,0,2,]])
    edge_addr = torch.Tensor([[1],[0],[1]])
    demo = Data(x=x,edge_index=edge_index,edge_attr = edge_addr)
    hgat = hgatConv(node_featrues_num=2)
    print(hgat( x = demo.x,edge_index = demo.edge_index,edge_attr = demo.edge_attr))
