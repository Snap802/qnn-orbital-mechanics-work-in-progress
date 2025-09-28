# graph_problem.py
import networkx as nx
import numpy as np

def create_toy_graph():
    G = nx.Graph()
    coords = {
        0: (0,0),
        1: (1,0.2),
        2: (2,0.0),
        3: (1.5,1.0),
        4: (0.5,1.2),
        5: (2.5,0.8)
    }
    for i,p in coords.items():
        G.add_node(i, pos=p)
    edges = [(0,1),(1,2),(0,4),(4,3),(3,5),(2,5),(1,3),(4,2)]
    for a,b in edges:
        pa = np.array(coords[a]); pb = np.array(coords[b])
        dist = np.linalg.norm(pa-pb)
        cost = dist + 0.1*np.random.rand()
        G.add_edge(a,b, weight=cost)
    return G
