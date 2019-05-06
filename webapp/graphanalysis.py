import os,sys
from pathlib import Path
import json
import array as arr
import networkx as nx
from networkx.algorithms import approximation as ap
import subprocess

def refine(path):
    for entry in os.scandir(path):
        if entry.name.split('.')[0] .isdigit():
            with open(entry,'rt') as jsonfile:#, open(entry.name.split('.')[0]+'.edgelist','w') as jf:
                data = json.load(jsonfile)
                # print(data["x"])
                G = nx.MultiDiGraph()#MultiDiGraph()DiGraph
                H = nx.DiGraph()
                for i in data["x"]:
                    # print(i[0])
                    G.add_node(i[0])
                    H.add_node(i[0])
                for i in data["edge_index"]:
                    # j = (i[0],i[1])
                    # print(j)
                    G.add_edge(i[0],i[1])
                    H.add_edge(i[0],i[1])
                I = H.to_undirected()
                ####MERGED##############for MuliDiGraph
                # if max(G.degree)[1]>3:
                #     subprocess.run(["cp",path+"/"+entry.name,"merge/"])
                ####CLUSTERING##########for DiGraph
                # if nx.average_clustering(G)>0.05:
                #     # print(nx.average_clustering(G))
                #     subprocess.run(["cp",path+"/"+entry.name,"cluster/"])
                #####CYCLE#############for Graph
                # if len(nx.cycle_basis(G))>10:
                #     subprocess.run(["cp",path+"/"+entry.name,"cycle/"])
                #####WIDTH#############for Graph
                # if ap.treewidth.treewidth_min_degree(G)[0]>3:
                    # subprocess.run(["cp",path+"/"+entry.name,"width/"])
                #print(ap.treewidth.treewidth_min_degree(G)[0],
                #ap.treewidth.treewidth_min_fill_in(G)[0])
                #####LENGTH############for MultiDiGraph
                # if len(max(nx.all_pairs_shortest_path_length(G))[1])>=10:
                    # subprocess.run(["cp",path+"/"+entry.name,"length/"])
                # print(len(max(nx.all_pairs_shortest_path_length(G))[1]))
                if (max(G.degree)[1]<=3) and (nx.average_clustering(H)<=0.05) and (len(nx.cycle_basis(I))<=10) and (ap.treewidth.treewidth_min_degree(I)[0]<=3) and (len(max(nx.all_pairs_shortest_path_length(G))[1])<10):
                    subprocess.run(["cp",path+"/"+entry.name,"simple/"])
                    # print(max(G.degree)[1],nx.average_clustering(H))

                # print(entry.name,max(G.degree)[1])#G.nodes, G.edges)



refine(sys.argv[1])