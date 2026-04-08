"""QEM-DQAS demo with Qiskit (density-matrix simulation) + PyTorch.

This is a PyTorch/Qiskit counterpart for `examples/QEM_DQAS.py`.
The search space is discrete (single-qubit placeholder gates), so we only
optimize structure logits via a REINFORCE-style estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, state_fidelity


@dataclass
class QEMConfig:
    # number of qubits for QFT/QEM task
    n: int = 4
    # DQAS optimization iterations
    epochs: int = 40
    # architecture samples per epoch
    batch: int = 24
    # optimizer learning rate for structure logits stp
    lr_structure: float = 0.2
    # bit-flip probability for idle qubits in each QFT moment
    p_idle: float = 0.2
    # global bit-flip probability after each QFT moment
    p_sep: float = 0.02
    # random seed (torch / numpy / random unitary design)
    seed: int = 7


def _x_channel(p: float) -> Kraus:
    """Single-qubit bit-flip channel E(rho)=(1-p)rho+p X rho X."""
    i2 = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return Kraus([np.sqrt(1 - p) * i2, np.sqrt(p) * x])


def unitary_design(n: int, l: int = 3, seed: int = 0) -> QuantumCircuit:
    """
    Prepare a random pre-circuit to mimic the original unitary-design prepend stage.

    This is a lightweight approximation of Haar-random state preparation and is used
    as the QEM reference-state generator before applying QFT/noisy-QFT branches.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)

    for q in range(n):
        qc.h(q)

    for _ in range(l):
        for q in range(n):
            theta = rng.choice([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
            qc.rz(theta, q)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    qc.cz(i, j)
        for q in range(n):
            qc.h(q)

    return qc


def qft_moments(n: int) -> List[List[Tuple[str, Tuple[int, ...], float]]]:
    """Build moments in the same logical order as the Cirq version."""
    moments: List[List[Tuple[str, Tuple[int, ...], float]]] = []
    for i in reversed(range(n)):
        moments.append([("h", (i,), 0.0)])
        for d, j in enumerate(reversed(range(i))):
            theta = np.pi / (2 ** (d + 1))
            moments.append([("cp", (j, i), theta)])
    return moments


def idle_slots(moments: Sequence[Sequence[Tuple[str, Tuple[int, ...], float]]], n: int) -> List[int]:
    """
    Enumerate all idle-qubit slots across moments.

    Each idle slot corresponds to one architecture choice in DQAS.
    Therefore, architecture length p == number of idle slots.
    """
    slots: List[int] = []
    for layer in moments:
        occ = set()
        for _, qubits, _ in layer:
            occ.update(qubits)
        for q in range(n):
            if q not in occ:
                slots.append(q)
    return slots


def apply_named_single_qubit(qc: QuantumCircuit, name: str, q: int) -> None:
    """Map a symbolic gate name from op_pool to a concrete Qiskit gate."""
    if name == "I":
        return
    if name == "X":
        qc.x(q)
    elif name == "Y":
        qc.y(q)
    elif name == "Z":
        qc.z(q)
    elif name == "H":
        qc.h(q)
    elif name == "S":
        qc.s(q)
    elif name == "T":
        qc.t(q)
    elif name == "RX_PI_3":
        qc.rx(np.pi / 3, q)
    elif name == "RX_2PI_3":
        qc.rx(2 * np.pi / 3, q)
    elif name == "RZ_PI_3":
        qc.rz(np.pi / 3, q)
    elif name == "RZ_2PI_3":
        qc.rz(2 * np.pi / 3, q)
    else:
        raise ValueError(f"Unknown gate {name}")


def append_layer(qc: QuantumCircuit, layer: Sequence[Tuple[str, Tuple[int, ...], float]]) -> None:
    for name, qubits, theta in layer:
        if name == "h":
            qc.h(qubits[0])
        elif name == "cp":
            qc.cp(theta, qubits[0], qubits[1])
        else:
            raise ValueError(f"Unknown layer op {name}")


def build_filled_qft_circuit(
    n: int,
    preset: Sequence[int],
    op_pool: Sequence[str],
    seed: int,
) -> QuantumCircuit:
    """
    Build a visualization-friendly circuit: prepend + QFT + DQAS placeholder gates.

    Note: noise channels are NOT drawn here (channels are applied on density matrix in
    `qem_loss`). This circuit is intended for architecture visualization only.
    """
    moments = qft_moments(n)
    slots = idle_slots(moments, n)
    if len(preset) != len(slots):
        raise ValueError(f"preset length {len(preset)} != slots length {len(slots)}")

    qc = unitary_design(n, l=3, seed=seed)
    k = 0
    for layer in moments:
        append_layer(qc, layer)
        occ = set()
        for _, qubits, _ in layer:
            occ.update(qubits)
        for q in range(n):
            if q not in occ:
                apply_named_single_qubit(qc, op_pool[preset[k]], q)
                k += 1
    return qc


def qem_loss(
    n: int,
    preset: Sequence[int],
    op_pool: Sequence[str],
    p_idle: float,
    p_sep: float,
    seed: int,
) -> float:
    """
    Compute QEM objective: negative fidelity between ideal and noisy branches.

    - ideal branch: prepend + clean QFT
    - noisy branch: prepend + (QFT + placeholder gates on idle slots) + noise channels
    """
    moments = qft_moments(n)
    slots = idle_slots(moments, n)

    if len(preset) != len(slots):
        raise ValueError(f"preset length {len(preset)} != slots length {len(slots)}")

    prepend = unitary_design(n, l=3, seed=seed)

    # ideal branch
    ideal = QuantumCircuit(n)
    ideal.compose(prepend, inplace=True)
    for layer in moments:
        append_layer(ideal, layer)
    dm_ideal = DensityMatrix.from_instruction(ideal)

    # noisy branch with placeholders + channels
    dm_noisy = DensityMatrix.from_instruction(prepend)
    x_idle = _x_channel(p_idle)
    x_sep = _x_channel(p_sep)

    k = 0
    for layer in moments:
        lqc = QuantumCircuit(n)
        append_layer(lqc, layer)

        occ = set()
        for _, qubits, _ in layer:
            occ.update(qubits)

        for q in range(n):
            if q not in occ:
                # Architecture decides which single-qubit gate to place on idle wire.
                gate_name = op_pool[preset[k]]
                apply_named_single_qubit(lqc, gate_name, q)
                k += 1

        dm_noisy = dm_noisy.evolve(Operator(lqc))

        for q in range(n):
            if q not in occ:
                # Idle noise channel
                dm_noisy = dm_noisy.evolve(x_idle, qargs=[q])
        for q in range(n):
            # Separation noise channel on all qubits
            dm_noisy = dm_noisy.evolve(x_sep, qargs=[q])

    fidelity = state_fidelity(dm_ideal, dm_noisy)
    return -float(fidelity)


def qem_fidelity(
    n: int,
    preset: Sequence[int],
    op_pool: Sequence[str],
    p_idle: float,
    p_sep: float,
    seed: int,
) -> float:
    """Convenience wrapper: fidelity = -qem_loss."""
    return -qem_loss(
        n=n,
        preset=preset,
        op_pool=op_pool,
        p_idle=p_idle,
        p_sep=p_sep,
        seed=seed,
    )


def train_qem_dqas(
    cfg: QEMConfig, op_pool: Sequence[str]
) -> Tuple[torch.Tensor, List[float], List[int], torch.Tensor]:
    """
    DQAS structure optimization for discrete operator pools.

    We optimize logits stp (shape [p, c]); each row is a categorical distribution
    over candidate gates for one idle slot.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    p = len(idle_slots(qft_moments(cfg.n), cfg.n))
    c = len(op_pool)
    stp = torch.nn.Parameter(0.01 * torch.randn(p, c))
    init_stp = stp.detach().clone()
    optimizer = torch.optim.Adam([stp], lr=cfg.lr_structure)

    history: List[float] = []

    for epoch in range(cfg.epochs):
        prob = torch.softmax(stp, dim=1)

        grad = torch.zeros_like(stp)
        losses: List[float] = []

        for b in range(cfg.batch):
            # Sample one candidate architecture from per-slot categorical distributions.
            idx = torch.multinomial(prob, num_samples=1).squeeze(-1)
            preset = idx.tolist()

            loss = qem_loss(
                n=cfg.n,
                preset=preset,
                op_pool=op_pool,
                p_idle=cfg.p_idle,
                p_sep=cfg.p_sep,
                seed=cfg.seed + b,
            )
            losses.append(loss)

            onehot = torch.zeros_like(stp)
            onehot[torch.arange(p), idx] = 1.0
            # Score-function term: grad log pi(a|stp) = onehot - prob
            grad += onehot - prob.detach()

        baseline = float(np.mean(losses))
        advantages = torch.tensor([l - baseline for l in losses], dtype=torch.float32)
        scale = advantages.abs().mean().item() + 1e-8

        # REINFORCE-style update with simple magnitude normalization.
        stp_grad = -(grad / cfg.batch) * scale

        optimizer.zero_grad()
        stp.grad = stp_grad
        optimizer.step()

        history.append(baseline)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            best = torch.argmax(prob, dim=1).tolist()
            print(f"epoch={epoch:03d} loss={baseline:.6f} p={p} c={c} best[:8]={best[:8]}")

    best_preset = torch.argmax(torch.softmax(stp, dim=1), dim=1).tolist()
    return stp.detach(), history, best_preset, init_stp


def save_visualizations(
    *,
    cfg: QEMConfig,
    op_pool: Sequence[str],
    init_stp: torch.Tensor,
    final_stp: torch.Tensor,
    best_preset: Sequence[int],
    out_dir: str = "qem_dqas_artifacts",
) -> None:
    """Save initial/final architecture visualization and parameter summaries."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    init_prob = torch.softmax(init_stp, dim=1)
    final_prob = torch.softmax(final_stp, dim=1)
    init_preset = torch.argmax(init_prob, dim=1).tolist()

    init_circuit = build_filled_qft_circuit(cfg.n, init_preset, op_pool, seed=cfg.seed)
    final_circuit = build_filled_qft_circuit(cfg.n, best_preset, op_pool, seed=cfg.seed)

    # Text visualization is robust and does not require extra plotting dependencies.
    (out / "initial_circuit.txt").write_text(str(init_circuit.draw(output="text")))
    (out / "filled_circuit_after_dqas.txt").write_text(str(final_circuit.draw(output="text")))

    # Save structure parameters (logits/probabilities) for both initial and final states.
    np.savetxt(out / "initial_stp_logits.csv", init_stp.numpy(), delimiter=",")
    np.savetxt(out / "final_stp_logits.csv", final_stp.numpy(), delimiter=",")
    np.savetxt(out / "initial_stp_prob.csv", init_prob.numpy(), delimiter=",")
    np.savetxt(out / "final_stp_prob.csv", final_prob.numpy(), delimiter=",")

    # Human-readable mapping from slot to selected gate and probability.
    lines = ["slot,init_gate,init_prob,final_gate,final_prob"]
    for i, gidx in enumerate(best_preset):
        init_idx = int(init_preset[i])
        lines.append(
            f"{i},{op_pool[init_idx]},{float(init_prob[i, init_idx]):.6f},"
            f"{op_pool[gidx]},{float(final_prob[i, gidx]):.6f}"
        )
    (out / "architecture_summary.csv").write_text("\n".join(lines))

    # Fidelity comparison: initial architecture vs DQAS-selected architecture.
    init_fid = qem_fidelity(
        n=cfg.n,
        preset=init_preset,
        op_pool=op_pool,
        p_idle=cfg.p_idle,
        p_sep=cfg.p_sep,
        seed=cfg.seed,
    )
    final_fid = qem_fidelity(
        n=cfg.n,
        preset=best_preset,
        op_pool=op_pool,
        p_idle=cfg.p_idle,
        p_sep=cfg.p_sep,
        seed=cfg.seed,
    )
    fidelity_report = (
        "metric,value\n"
        f"initial_fidelity,{init_fid:.10f}\n"
        f"post_dqas_fidelity,{final_fid:.10f}\n"
        f"fidelity_improvement,{(final_fid - init_fid):.10f}\n"
    )
    (out / "fidelity_report.csv").write_text(fidelity_report)

    print(f"[visualization] saved to: {out.resolve()}")
    print(f"[visualization] initial circuit: {out / 'initial_circuit.txt'}")
    print(f"[visualization] final circuit:   {out / 'filled_circuit_after_dqas.txt'}")
    print(f"[fidelity] initial={init_fid:.8f}, post_dqas={final_fid:.8f}, delta={final_fid - init_fid:+.8f}")


def main_3() -> None:
    op_pool = ["X", "Y", "Z", "I", "T", "S"]
    cfg = QEMConfig(n=3, epochs=30, batch=32)
    stp, history, best, init_stp = train_qem_dqas(cfg, op_pool)
    print("\n[QEM n=3] best architecture length:", len(best))
    print("[QEM n=3] last-5 history:", history[-5:])
    save_visualizations(cfg=cfg, op_pool=op_pool, init_stp=init_stp, final_stp=stp, best_preset=best)


def main_4() -> None:
    op_pool = ["I", "X", "Y", "Z", "H", "RX_PI_3", "RX_2PI_3", "RZ_PI_3", "RZ_2PI_3", "S", "T"]
    cfg = QEMConfig(n=4, epochs=20, batch=24)
    stp, history, best, init_stp = train_qem_dqas(cfg, op_pool)
    print("\n[QEM n=4] best architecture length:", len(best))
    print("[QEM n=4] last-5 history:", history[-5:])
    save_visualizations(cfg=cfg, op_pool=op_pool, init_stp=init_stp, final_stp=stp, best_preset=best)


if __name__ == "__main__":
    main_4()
