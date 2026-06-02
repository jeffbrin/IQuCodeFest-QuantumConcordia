from qiskit import QuantumCircuit, QuantumRegister
from typing import List

from utils import int_to_little_endian_bits, apply_x_for_zero_controls


def apply_controlled_x_on_basis_state(
    circuit: QuantumCircuit,
    controls: List,
    basis_value: int,
    target,
) -> None:
    """
    Apply X to target if the control register equals basis_value.

    The control register is interpreted little-endian.

    This implements:
        if controls == basis_value:
            target ^= 1
    """
    width = len(controls)
    pattern = int_to_little_endian_bits(basis_value, width)

    apply_x_for_zero_controls(circuit, controls, pattern)

    if width == 0:
        circuit.x(target)
    elif width == 1:
        circuit.cx(controls[0], target)
    else:
        circuit.mcx(controls, target)

    apply_x_for_zero_controls(circuit, controls, pattern)


def apply_qrom_lookup(
    circuit: QuantumCircuit,
    index_register: QuantumRegister,
    output_register: QuantumRegister,
    chunks: List[List[int]],
) -> None:
    """
    Reversibly load a chunk into output_register.

    Implements:
        |idx>|y> -> |idx>|y XOR chunk_idx>

    If output_register starts as |0...0>, the result is:
        |idx>|chunk_idx>

    For a small demo, this is implemented as a hard-coded qROM:
        for every possible idx,
            for every output bit that is 1 in chunk_idx,
                apply controlled-X on that output bit.

    This is not hardware efficient, but it is conceptually explicit.
    """
    index_qubits = list(index_register)

    for idx, chunk_bits in enumerate(chunks):
        for bit_pos, bit in enumerate(chunk_bits):
            if bit == 1:
                apply_controlled_x_on_basis_state(
                    circuit=circuit,
                    controls=index_qubits,
                    basis_value=idx,
                    target=output_register[bit_pos],
                )