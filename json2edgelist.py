import sys
import os
from pathlib import Path
import json
import torch
from torch_geometric.data import Data,InMemoryDataset,DataLoader




def datalist(path):
    data_list = []
    for entry in sorted(os.scandir(path), key=lambda x: (x.is_dir(), x.name)):
        if entry.name.split('.')[0] .isdigit():
            with open(entry,'rt') as jsonfile:#, open(entry.name.split('.')[0]+'.edgelist','w') as jf:
                jsons = json.load(jsonfile)
                # print(jsons["edges"])
                edge_index = torch.tensor(jsons['edges'])#,dtype=torch.long)
                data = Data(edge_index=edge_index.t().contiguous())
                # print(entry.name.split('.')[0],data)
                data_list.append(data)
    return data_list

data_list = datalist(sys.argv[1])
# datalist(sys.argv[1])
loader = DataLoader(data_list,batch_size = 32)
print(loader.get(0))
# print(dataset)
# data = dataset[0]
# model = GAE(encoder=lambda x: x)
# model.reset_parameters()
# edge_index = data.edge_index
# print(data,edge_index)

# model = GAE(encoder=lambda x: x)
# model.reset_parameters()

# class MyOwnDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(MyOwnDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['some_file_1', 'some_file_2', ...]

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         pass
#         # Download to `self.raw_dir`.

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = datalist(sys.argv[1])

#         if self.pre_filter is not None:
#             data_list [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])




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