"""DQAS demo with Qiskit (circuit simulation) + PyTorch (optimization).

This script is a replacement-style example for the original TensorFlow-based
DQAS examples in this repository. It keeps the spirit of DQAS:

1. Maintain a probabilistic structure parameter ``stp`` over an operator pool.
2. Maintain operator parameters ``nnp`` for each layer-candidate pair.
3. Sample candidate architectures and optimize both ``stp`` and ``nnp``.

The quantum circuit execution is implemented with Qiskit's ``Statevector`` and
the optimization loop is implemented with PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class DQASConfig:
    n: int = 6
    d: int = 3
    p: int = 6
    c: int = 4
    epochs: int = 80
    batch: int = 12
    seed: int = 7
    lr_structure: float = 0.15
    lr_network: float = 0.05


def regular_graph_generator(n: int, d: int, seed: int) -> nx.Graph:
    return nx.random_regular_graph(d=d, n=n, seed=seed)


def softmax_prob(stp: torch.Tensor) -> torch.Tensor:
    return torch.softmax(stp, dim=1)


def sample_preset(prob: torch.Tensor) -> List[int]:
    # one categorical sample for each layer
    preset = torch.multinomial(prob, num_samples=1).squeeze(-1)
    return preset.tolist()


def build_circuit(n: int, params: Sequence[float], preset: Sequence[int], g: nx.Graph) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)

    for layer, op in enumerate(preset):
        theta = float(params[layer])

        # operator pool (4 choices)
        # 0: RX on all qubits
        if op == 0:
            for q in range(n):
                qc.rx(theta, q)
        # 1: RY on all qubits
        elif op == 1:
            for q in range(n):
                qc.ry(theta, q)
        # 2: RZZ on graph edges
        elif op == 2:
            for u, v in g.edges:
                qc.rzz(theta, u, v)
        # 3: Mixer-like RX with alternating sign
        elif op == 3:
            for q in range(n):
                qc.rx(theta if (q % 2 == 0) else -theta, q)
        else:
            raise ValueError(f"Unknown op index {op}")

    return qc


def maxcut_value_from_statevector(state: Statevector, g: nx.Graph) -> float:
    probs = np.abs(state.data) ** 2
    n = g.number_of_nodes()
    value = 0.0

    for bitstr, p in enumerate(probs):
        # qiskit little-endian bit-order
        bits = [(bitstr >> i) & 1 for i in range(n)]
        cut = 0
        for u, v in g.edges:
            if bits[u] != bits[v]:
                cut += 1
        value += p * cut

    return float(value)


def objective(n: int, params: Sequence[float], preset: Sequence[int], g: nx.Graph) -> float:
    qc = build_circuit(n, params, preset, g)
    state = Statevector.from_instruction(qc)
    # maximize cut <=> minimize negative cut
    return -maxcut_value_from_statevector(state, g)


def parameter_shift_gradient(
    n: int,
    params: torch.Tensor,
    preset: Sequence[int],
    g: nx.Graph,
) -> torch.Tensor:
    grads = torch.zeros_like(params)
    shift = np.pi / 2

    base = params.detach().cpu().numpy()
    for i in range(len(base)):
        plus = base.copy()
        minus = base.copy()
        plus[i] += shift
        minus[i] -= shift
        f_plus = objective(n, plus, preset, g)
        f_minus = objective(n, minus, preset, g)
        grads[i] = 0.5 * (f_plus - f_minus)

    return grads


def train_dqas_qiskit_torch(cfg: DQASConfig) -> Tuple[torch.Tensor, torch.Tensor, List[float], nx.Graph]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    g = regular_graph_generator(cfg.n, cfg.d, cfg.seed)

    # structure params (logits) and network params
    stp = torch.nn.Parameter(0.01 * torch.randn(cfg.p, cfg.c))
    nnp = torch.nn.Parameter(0.01 * torch.randn(cfg.p, cfg.c))

    opt_s = torch.optim.Adam([stp], lr=cfg.lr_structure)
    opt_n = torch.optim.Adam([nnp], lr=cfg.lr_network)

    history: List[float] = []

    for epoch in range(cfg.epochs):
        prob = softmax_prob(stp)

        losses = []
        gs_acc = torch.zeros_like(stp)
        gnnp_acc = torch.zeros_like(nnp)

        for _ in range(cfg.batch):
            preset = sample_preset(prob)
            idx = torch.tensor(preset, dtype=torch.long)
            layers = torch.arange(cfg.p, dtype=torch.long)
            selected_params = nnp[layers, idx]

            loss = objective(cfg.n, selected_params.detach().cpu().numpy(), preset, g)
            losses.append(loss)

            # network gradient via parameter-shift on selected parameters
            g_sel = parameter_shift_gradient(cfg.n, selected_params, preset, g)
            for layer in range(cfg.p):
                gnnp_acc[layer, preset[layer]] += g_sel[layer]

            # REINFORCE-style structure gradient (with epoch baseline)
            onehot = torch.zeros(cfg.p, cfg.c)
            onehot[layers, idx] = 1.0
            gs_acc += onehot - prob.detach()

        baseline = float(np.mean(losses))
        reinforce_weights = torch.tensor([l - baseline for l in losses], dtype=torch.float32)

        # aggregate structure gradient
        # use average policy score multiplied by average advantage scale
        scale = reinforce_weights.abs().mean().item() + 1e-8
        stp_grad = -(gs_acc / cfg.batch) * scale
        nnp_grad = gnnp_acc / cfg.batch

        opt_s.zero_grad()
        opt_n.zero_grad()
        stp.grad = stp_grad
        nnp.grad = nnp_grad
        opt_s.step()
        opt_n.step()

        history.append(baseline)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            best_arch = torch.argmax(prob, dim=1).tolist()
            print(f"epoch={epoch:03d} loss={baseline:.6f} best_arch={best_arch}")

    return stp.detach(), nnp.detach(), history, g


def main() -> None:
    cfg = DQASConfig()
    stp, nnp, history, g = train_dqas_qiskit_torch(cfg)

    prob = softmax_prob(stp)
    best_preset = torch.argmax(prob, dim=1).tolist()
    best_params = nnp[torch.arange(cfg.p), torch.tensor(best_preset)]
    best_loss = objective(cfg.n, best_params.numpy(), best_preset, g)

    print("\n=== Final Result ===")
    print("best architecture:", best_preset)
    print("best loss:", float(best_loss))
    print("best expected cut:", float(-best_loss))
    print("last-5 history:", history[-5:])


if __name__ == "__main__":
    main()
