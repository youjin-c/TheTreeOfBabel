import sys
import os
from pathlib import Path
import json

import torch
from torch_geometric.data import Data,InMemoryDataset,DataLoader
from torch import Tensor as T
from torch_geometric.nn import GAE, VGAE, ARGA, ARGVA

def datalist(path):
    data_list = []
    for entry in sorted(os.scandir(path), key=lambda x: (x.is_dir(), x.name)):
        if entry.name.split('.')[0] .isdigit():
            with open(entry,'rt') as jsonfile:
                jsons = json.load(jsonfile)
                x = torch.tensor(jsons['x'])
                edge_index = torch.tensor(jsons['edge_index'])#,dtype=torch.long)
                data = Data(x=x, edge_index=edge_index.t().contiguous())# print(entry.name.split('.')[0],data)
                data_list.append(data)
    return data_list

# datalist(sys.argv[1])
data_list = datalist(sys.argv[1])
# print(data_list[0].edge_index)




loader = DataLoader(data_list,batch_size = 32,shuffle=False)
# for data in loader: #batch,
#     print(data)
#     print(data.x)
#     print(data.edge_index)


# data = dataset[0]
model = GAE(encoder=lambda x: x)
model.reset_parameters()
for data in loader:
    z = model.encode(data.x)
    adj = model.decoder.forward_all(z)
    value = model.decode(z, data.edge_index)
    print(value)
    # print(data)
    # print(data.x)
    # print(data.edge_index)


def test_gae():
    model = GAE(encoder=lambda x: x)
    model.reset_parameters()

    x = torch.Tensor([[1, -1], [1, 2], [2, 1]])
    z = model.encode(x)
    assert z.tolist() == x.tolist()

    adj = model.decoder.forward_all(z)
    assert adj.tolist() == torch.sigmoid(
        torch.Tensor([[+2, -1, +1], [-1, +5, +4], [+1, +4, +5]])).tolist()
    # print("adj",adj.tolist())

    edge_index = torch.tensor([[0, 1], [1, 2]])
    value = model.decode(z, edge_index)
    assert value.tolist() == torch.sigmoid(torch.Tensor([-1, 4])).tolist()
    # print("value",value.tolist())

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    data = Data(edge_index=edge_index)
    data.num_nodes = edge_index.max().item() + 1
    data = model.split_edges(data, val_ratio=0.2, test_ratio=0.3)

    assert data.val_pos_edge_index.size() == (2, 2)
    assert data.val_neg_edge_index.size() == (2, 2)
    assert data.test_pos_edge_index.size() == (2, 3)
    assert data.test_neg_edge_index.size() == (2, 3)
    assert data.train_pos_edge_index.size() == (2, 5)
    assert data.train_neg_adj_mask.size() == (11, 11)
    assert data.train_neg_adj_mask.sum().item() == (11**2 - 11) / 2 - 4 - 6 - 5

    z = torch.randn(11, 16)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    assert loss.item() > 0

    auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    assert auc >= 0 and auc <= 1 and ap >= 0 and ap <= 1