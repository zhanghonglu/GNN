import torch
import os
import torch.nn.functional as F
from covid_data.subGraph import get_subGraph
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE,GATConv
from model.utils.split_edge import train_test_split_edges
from model.hgate import hgat_Encoder
from model.LinearRegression import LinearRegression
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_orign = get_subGraph(1, "New York").to(dev)
torch.manual_seed(12345)

# data_orign.train_mask = data_orign.val_mask = data_orign.test_mask  = None
# data = train_test_split_edges(data_orign)
model = GAE(hgat_Encoder(node_featrues_num=5)).to(dev).double()
risk_model = LinearRegression(feature_num=5).to(dev).double()
print("加载模型...")
model.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'parameter', str(26000001)+'parameter.pkl')))
risk_model.load_state_dict(torch.load(os.path.join(os.path.abspath('.'), 'parameter', 'epoch10000risk_model_parameter.pkl')))
# x, train_pos_edge_index, train_pos_edge_addr= data.x.double().to(dev), data.train_pos_edge_index.to(dev), data.train_pos_edge_attr.to(dev)
# optimizer = torch.optim.Adam(model.parameters(),  lr=0.05)


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

def risk_eval_train():
    risk_model.train()
    z = model.encode(data_orign.x, data_orign.edge_index, data_orign.edge_attr)
    y_eval = risk_model(z)
    optimizer_risk = torch.optim.SGD(risk_model.parameters(), lr=0.0001)
    loss_risk_function = torch.nn.MSELoss()
    loss_risk = loss_risk_function(y_eval, data_orign.y)
    print("risk_loss:{}".format(loss_risk))

    optimizer_risk.zero_grad()
    loss_risk.backward()
    optimizer_risk.step()





for epoch in range(1, 1000000):
    risk_eval_train()
    # train()
    print("epoch{}".format(epoch) .format())
    if epoch%10000==0:
        torch.save(risk_model.state_dict(), os.path.join(os.path.abspath('.'), 'parameter', 'epoch'+str(epoch)+'risk_model_parameter.pkl'))
    # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    # print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))





