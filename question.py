import sys
import os
from pathlib import Path
import json
import torch
import argparse
from torch_geometric.data import Data,InMemoryDataset,DataLoader
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
dataSel = data_list[0]

#dataSel == {"x": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116]], "edge_index": [[2, 3, 3, 4, 3, 5, 4, 8, 4, 10, 5, 6, 5, 16, 6, 22, 6, 114, 8, 9, 9, 17, 9, 18, 10, 20, 10, 61, 10, 21, 11, 12, 11, 13, 12, 66, 12, 78, 13, 14, 13, 15, 16, 24, 16, 25, 17, 26, 17, 28, 18, 19, 19, 29, 19, 30, 20, 93, 21, 33, 21, 35, 22, 23, 23, 36, 23, 37, 24, 38, 24, 40, 25, 43, 25, 41, 25, 44, 25, 45, 26, 27, 27, 46, 27, 47, 28, 51, 29, 52, 29, 53, 30, 54, 30, 92, 32, 58, 32, 64, 33, 34, 34, 68, 34, 79, 35, 76, 35, 80, 36, 11, 37, 11, 38, 39, 39, 82, 39, 87, 40, 115, 40, 83, 47, 48, 48, 49, 48, 50, 49, 25, 50, 81, 54, 55, 55, 94, 55, 98, 55, 107, 56, 57, 56, 60, 58, 59, 59, 62, 59, 63, 61, 104, 61, 106, 64, 65, 65, 102, 66, 70, 66, 74, 70, 71, 71, 72, 71, 103, 79, 86, 79, 88, 83, 85, 83, 84, 88, 89, 89, 90, 89, 91, 92, 56, 93, 31, 93, 32, 94, 96, 94, 100, 96, 97, 104, 105, 105, 107, 105, 109, 106, 111, 106, 113, 109, 110, 111, 112, 114, 61], [3, 2, 4, 3, 5, 3, 8, 4, 10, 4, 6, 5, 16, 5, 22, 6, 114, 6, 9, 8, 17, 9, 18, 9, 20, 10, 61, 10, 21, 10, 12, 11, 13, 11, 66, 12, 78, 12, 14, 13, 15, 13, 24, 16, 25, 16, 26, 17, 28, 17, 19, 18, 29, 19, 30, 19, 93, 20, 33, 21, 35, 21, 23, 22, 36, 23, 37, 23, 38, 24, 40, 24, 43, 25, 41, 25, 44, 25, 45, 25, 27, 26, 46, 27, 47, 27, 51, 28, 52, 29, 53, 29, 54, 30, 92, 30, 58, 32, 64, 32, 34, 33, 68, 34, 79, 34, 76, 35, 80, 35, 11, 36, 11, 37, 39, 38, 82, 39, 87, 39, 115, 40, 83, 40, 48, 47, 49, 48, 50, 48, 25, 49, 81, 50, 55, 54, 94, 55, 98, 55, 107, 55, 57, 56, 60, 56, 59, 58, 62, 59, 63, 59, 104, 61, 106, 61, 65, 64, 102, 65, 70, 66, 74, 66, 71, 70, 72, 71, 103, 71, 86, 79, 88, 79, 85, 83, 84, 83, 89, 88, 90, 89, 91, 89, 56, 92, 31, 93, 32, 93, 96, 94, 100, 94, 97, 96, 105, 104, 107, 105, 109, 105, 111, 106, 113, 106, 110, 109, 112, 111, 61, 114]]}


loader = DataLoader(data_list,batch_size = len(data_list),shuffle=False)
for batch in loader:
    # print(batch.num_features)
    # print(batch.num_graphs)
    pass


data = Data(x=dataSel.x, edge_index=dataSel.edge_index)

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
x, edge_index = data.x.to(device), data.edge_index.to(device)###
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + 0.001 * model.kl_loss()
    loss.backward()
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, 101):
    train()
    auc, ap = test(data.val_pos_edge_index, data.val_neg_edge_index)
    # print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


z = model.encode(x,edge_index)
value = model.decode(z, edge_index)
value_list= value.tolist()

