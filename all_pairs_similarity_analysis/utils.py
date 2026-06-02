def int_to_little_endian_bits(value: int, width: int) -> List[int]:
    """
    Convert an integer to a list of little-endian bits.

    Example:
        value = 6 = binary 110
        width = 3
        returns [0, 1, 1]
    """
    return [(value >> k) & 1 for k in range(width)]

# ---------------------------------------------------------------------------
# Multi-controlled phase and qROM helpers
# ---------------------------------------------------------------------------

def apply_x_for_zero_controls(
    circuit: QuantumCircuit,
    controls: List,
    bit_pattern: List[int],
) -> None:
    """
    Convert controls-on-0 into controls-on-1 by applying X gates.

    If bit_pattern[k] == 0, then control qubit k is supposed to be |0>.
    Qiskit's mcx controls on |1>, so we wrap that qubit with X gates.
    """
    for qubit, bit in zip(controls, bit_pattern):
        if bit == 0:
            circuit.x(qubit)

