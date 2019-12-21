# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# import torch
# from torch_geometric.data import Data
#
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1,3], [0,4], [1,2]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index, y=[3,[1,0,4]])
#
# data
#
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index.t().contiguous())
# data
#
# data.edge_index
#
# for key, item in data:
#     print("{} found in data".format(key))
#
# data.num_nodes
#
# data.num_edges
#
# from torch_geometric.datasets import TUDataset
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
#
# dataset.num_classes
#
# len(dataset)a
#
# for graph in dataset:
#     print(graph)
#
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
#
# len(dataset)
#
# dataset.num_classes
#
# data = dataset[0]
#
# len(data.train_mask)
#
# data
#
# data.val_mask.sum().item()
#
# data.test_mask.sum().item()
#
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
#
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# len(dataset)
#
# 600%32
#
# for batch in loader:
#     print(batch)
#     #>>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])
#     print(batch.num_graphs)
#
# from torch_scatter import scatter_mean
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
#
# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for data in loader:
#     print(data)
#     #>>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])
#
#     print(data.num_graphs)
#     #>>> 32
#
#     x = scatter_mean(data.x, data.batch, dim=0)
#     print(x.size())
#     
#     print(data.num_features)
#     #>>> torch.Size([32, 21])
#
# from torch_geometric.datasets import ShapeNet
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
# dataset[0]
#
# import torch_geometric.transforms as T
#
# dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
#                     pre_transform=T.KNNGraph(k=6))
#
# dataset[0]

# from torch_geometric.datasets import Planetoid
#
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(dataset.num_node_features, 16)
        #self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(x.shape)
        exit(0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# +
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1):
    #optimizer.zero_grad()
    out = model(data)
    exit(0)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss.backward()
    #optimizer.step()
# -

data


