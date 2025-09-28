# qaoa_solver.py
import pennylane as qml
from pennylane import numpy as np

def qaoa_cost_from_edges(edges, weights):
    H = []
    coeffs = []
    ops = []
    for (i,j), w in zip(edges, weights):
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        coeffs.append(w)
    return ops, coeffs

class QAOASolver:
    def __init__(self, n_qubits=6, p=1, seed=0):
        self.n_qubits = n_qubits
        self.p = p
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        rng = np.random.RandomState(seed)
        self.gamma = np.array(rng.uniform(0, np.pi, size=p), requires_grad=True)
        self.beta = np.array(rng.uniform(0, np.pi, size=p), requires_grad=True)

    def circuit(self, gammas, betas, cost_h_ops=None, cost_coeffs=None):
        @qml.qnode(self.dev, interface='autograd')
        def qnode(gammas, betas):
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            for layer in range(self.p):
                for (op, coeff) in zip(cost_h_ops, cost_coeffs):
                    pass
                for w in range(self.n_qubits):
                    qml.RX(2*betas[layer], wires=w)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return qnode

    def optimize(self, cost_h_ops, cost_coeffs, steps=30):
        opt = qml.AdamOptimizer(0.2)
        params = np.concatenate([self.gamma, self.beta])
        def closure(p):
            g = p[:self.p]
            b = p[self.p:]
            qnode = self.circuit(g,b,cost_h_ops,cost_coeffs)
            expvals = qnode(g,b)
            return np.sum(expvals)
        for s in range(steps):
            params = opt.step(closure, params)
        return params
