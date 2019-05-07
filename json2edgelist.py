import sys
import os
from pathlib import Path
import json

import torch
from torch_geometric.data import Data,InMemoryDataset,DataLoader

import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE


def datalist(pathTorch):
    data_list = []
    filename_list = []
    for entry in sorted(os.scandir(pathTorch), key=lambda x: (x.is_dir(), x.name)):
        if entry.name.split('.')[0] .isdigit():
            with open(entry,'rt') as jsonfile:
                jsons = json.load(jsonfile)
                x = torch.tensor(jsons['x'], dtype=torch.float)
                edge_index = torch.tensor(jsons['edge_index'],dtype=torch.long)
                data = Data(x=x, edge_index=edge_index)# print(entry.name.split('.')[0],data)
                data_list.append(data)
                filename_list.append(entry.name)
    return data_list,filename_list

path = './datasetTorch/basic/'
data_list,filename_list = datalist(path)
# print("datalen",len(data_list),"filename",filename_list)
# print(data_list[0].edge_index)
dataSel = data_list[0]

# print(data_list[0],filename_list[0])
# print(data)


loader = DataLoader(data_list,batch_size = len(data_list),shuffle=False)
# for data in loader: #batch,
#     print(data)
#     print(data.x)
#     print(data.edge_index)
for batch in loader:
    # print(batch.num_features)
    # print(batch.num_graphs)
    pass


data = Data(x=dataSel.x, edge_index=dataSel.edge_index)
# print(data.num_features)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
args = parser.parse_args()
assert args.model in ['GAE', 'VGAE']
kwargs = {'GAE': GAE, 'VGAE': VGAE}


class Encoder(torch.nn.Module):
    # def __init__(self, in_channels, out_channels):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# print(data.edge_index)

channels = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = kwargs[args.model](Encoder(batch.num_features, channels)).to(device)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = model.split_edges(data)
# edgelist=data['edge_index'].t()
x, train_pos_edge_index = data.x.to(device), data.train_pos_edge_index.to(device)###
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes)  * model.kl_loss()
    loss.backward()
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 401):
    train()
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    # print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


z = model.encode(x,train_pos_edge_index)
value = model.decode(z, train_pos_edge_index).tolist()
train_pos_edge_index = train_pos_edge_index.t()

decoded ={}
decoded['edge_index']=[]
decoded['x']=[]
for i, prob in enumerate(value):
    if prob >0.5:
        # print(i, prob, train_pos_edge_index[i].tolist())
        decoded['edge_index'].append(train_pos_edge_index[i].tolist())
        decoded['x'].append(train_pos_edge_index[i].tolist()[0])
        decoded['x'].append(train_pos_edge_index[i].tolist()[1])

decoded['x'] = list(dict.fromkeys(decoded['x']))

print(decoded)