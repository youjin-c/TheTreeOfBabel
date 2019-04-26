import glob
import sys
import re
import os
from pathlib import Path
import json
from collections import Counter
import array as arr
import torch
from torch_geometric.data import Data

def refine(path):
    i = 0
    for entry in sorted(os.scandir(path), key=lambda x: (x.is_dir(), x.name)):
        if entry.name.split('.')[0] .isdigit():
            with open(entry,'rt') as jsonfile:#, open(entry.name.split('.')[0]+'.edgelist','w') as jf:
                jsons = json.load(jsonfile)
                # print(jsons["edges"])
                edge_index = torch.tensor(jsons['edges'])#,dtype=torch.long)
                data = Data(edge_index=edge_index.t().contiguous())
                print(entry.name.split('.')[0],data)
                # i+=1
                # for i, line in enumerate(fd):
                    # for s in line.split():  
                        # if not s.isdigit():
                            # print(entry.name.split('.')[0],i,':',line)
                    #############cut off more than third column ######
                    # if len(line.split())>2:
                        # print(entry.name,i,':',line,re.findall(r"[\w\d']+", line))
                    #    print(entry.name,i,':',line,line.split(' ')[0:2])
                    #    jf.write(line.rsplit('\t',1)[0])
                    ##################################################
                        
                    # if not re.match("^[\d\s]+$",line): #[^0-9] #^[\d\s]+$ :numbers and space
                        # print(entry.name,i,':',line)
                        # jf.write(line.replace("E","1000"))
                    # else:
                        # pass
                        # jf.write(line)

refine(sys.argv[1])