# run_demo.py
import networkx as nx #type:ignore
import matplotlib.pyplot as plt
from graph_problem import create_toy_graph # type: ignore
from qaoa_solver import QAOASolver #type: ignore

def classical_baseline(G, source=0, target=5):
    path = nx.shortest_path(G, source=source, target=target, weight='weight')
    cost = sum(G[u][v]['weight'] for u,v in zip(path[:-1], path[1:]))
    return path, cost

def plot_path(G, path, fname='path.png'):
    pos = nx.get_node_attributes(G,'pos')
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, with_labels=True)
    edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G,pos, edgelist=edges, width=3, edge_color='r')
    plt.savefig(fname)
    plt.close()

if __name__=='__main__':
    G = create_toy_graph()
    p,c = classical_baseline(G)
    print('Classical shortest path', p, 'cost', c)
    plot_path(G,p,'classical_path.png')
    print('Saved classical_path.png')
    solver = QAOASolver(n_qubits=6,p=1)
    print('Running parameter sweep demo (not full QAOA mapping)...')
