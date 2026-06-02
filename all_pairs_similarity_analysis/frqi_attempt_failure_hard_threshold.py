#!/usr/bin/env python3
"""
FRQI-style quantum similarity Grover demo.

Goal
----
Given an image and a chunk size, split the image into non-overlapping chunks.
Search over ordered chunk pairs |i>|j> and amplify pairs whose FRQI-encoded
chunk states are similar.

This version DOES NOT precompute pairwise similarities to build the oracle.

Instead, for each Grover oracle call, it:

    1. Prepares |C_i> and |C_j> as FRQI-style chunk states.
    2. Runs a coherent swap-test-style phase oracle.
    3. Cancels trivial self-pairs i == j.
    4. Uncomputes the FRQI chunk states.
    5. Applies Grover diffusion over |i>|j>.

FRQI-style chunk state
----------------------
For a chunk with P pixels, define:

    |C_i> = 1/sqrt(P) sum_t |t>(
                cos(theta_{i,t})|0> + sin(theta_{i,t})|1>
            )

where:
    t indexes pixel position inside the chunk
    theta_{i,t} = pixel_intensity * pi/2

The swap test has:

    Pr(ancilla = 0) = (1 + |<C_i|C_j>|^2) / 2

This script uses that coherently as a soft phase oracle.

Important limitation
--------------------
This is not a hard-threshold oracle of the form:

    mark iff similarity >= threshold

A hard threshold requires coherent amplitude estimation or phase estimation.
This script is the smallest honest version that computes FRQI similarity
inside the quantum circuit rather than classically precomputing marked pairs.

Example
-------
    python frqi_quantum_similarity_grover.py \
        --image all_pairs_similarity_analysis/test.png \
        --chunk-size 16 \
        --iterations 1 \
        --shots 2048
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

from visualization import highlight_matching_chunks


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def is_power_of_two(x: int) -> bool:
    """Return True iff x is a positive power of two."""
    return x > 0 and (x & (x - 1)) == 0


def int_to_little_endian_bits(value: int, width: int) -> List[int]:
    """Convert integer to little-endian bit list."""
    return [(value >> k) & 1 for k in range(width)]


def bits_to_int_little_endian(bits: List[int]) -> int:
    """Convert little-endian bit list to integer."""
    value = 0
    for k, bit in enumerate(bits):
        value |= int(bit) << k
    return value


# ---------------------------------------------------------------------------
# Image loading and chunking
# ---------------------------------------------------------------------------

def load_grayscale_image(image_path: str) -> np.ndarray:
    """
    Load image as grayscale values in [0, 1].

    Returns
    -------
    image:
        NumPy array of shape (height, width), dtype float64.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def split_into_nonoverlapping_chunks(
    image: np.ndarray,
    chunk_size: int,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Split image into non-overlapping square chunks.

    Returns
    -------
    chunks:
        List of chunk arrays.

    positions:
        positions[i] = (row, col), the top-left coordinate of chunk i.

    cropped_shape:
        Shape of the portion of the image actually used.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    height, width = image.shape

    usable_height = (height // chunk_size) * chunk_size
    usable_width = (width // chunk_size) * chunk_size

    if usable_height == 0 or usable_width == 0:
        raise ValueError("Image is smaller than one chunk.")

    cropped = image[:usable_height, :usable_width]

    chunks: List[np.ndarray] = []
    positions: List[Tuple[int, int]] = []

    for row in range(0, usable_height, chunk_size):
        for col in range(0, usable_width, chunk_size):
            chunks.append(cropped[row:row + chunk_size, col:col + chunk_size].copy())
            positions.append((row, col))

    return chunks, positions, (usable_height, usable_width)


def chunks_to_angles(chunks: List[np.ndarray]) -> np.ndarray:
    """
    Convert chunks into FRQI color angles.

    For pixel intensity x in [0, 1]:

        theta = x * pi/2

    Ry(2 theta)|0> = cos(theta)|0> + sin(theta)|1>.

    Returns
    -------
    angles:
        Array of shape (num_chunks, pixels_per_chunk).
    """
    flat_chunks = [chunk.flatten(order="C") for chunk in chunks]
    return np.asarray(flat_chunks, dtype=np.float64) * (math.pi / 2.0)


# ---------------------------------------------------------------------------
# Controlled operations
# ---------------------------------------------------------------------------

def apply_x_for_zero_controls(
    circuit: QuantumCircuit,
    controls: List,
    bit_pattern: List[int],
) -> None:
    """
    Convert controls-on-0 into controls-on-1 by temporarily applying X gates.
    """
    for q, bit in zip(controls, bit_pattern):
        if bit == 0:
            circuit.x(q)


def apply_controlled_ry_on_basis_state(
    circuit: QuantumCircuit,
    controls: List,
    basis_bits_little_endian: List[int],
    target,
    angle: float,
) -> None:
    """
    Apply Ry(angle) to target iff controls match a basis state.

    The controls are interpreted little-endian.

    This is used to prepare FRQI color amplitudes:

        Ry(2 theta)|0> = cos(theta)|0> + sin(theta)|1>
    """
    if len(controls) != len(basis_bits_little_endian):
        raise ValueError("controls and basis_bits_little_endian must have same length.")

    if abs(angle) < 1e-12:
        return

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)

    if len(controls) == 0:
        circuit.ry(angle, target)
    else:
        controlled_ry = RYGate(angle).control(len(controls))
        circuit.append(controlled_ry, controls + [target])

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)


def apply_phase_flip_on_basis_state(
    circuit: QuantumCircuit,
    controls: List,
    basis_bits_little_endian: List[int],
) -> None:
    """
    Apply a -1 phase to exactly one computational basis state.
    """
    if len(controls) != len(basis_bits_little_endian):
        raise ValueError("controls and basis_bits_little_endian must have same length.")

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)

    n = len(controls)

    if n == 0:
        circuit.global_phase += math.pi
    elif n == 1:
        circuit.z(controls[0])
    else:
        target = controls[-1]
        ctrl = controls[:-1]
        circuit.h(target)
        circuit.mcx(ctrl, target)
        circuit.h(target)

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)


def apply_phase_flip_on_zero(
    circuit: QuantumCircuit,
    qubit,
) -> None:
    """
    Apply -1 phase iff qubit is |0>.

    X-Z-X implements:
        |0> -> -|0>
        |1> ->  |1>
    """
    circuit.x(qubit)
    circuit.z(qubit)
    circuit.x(qubit)


# ---------------------------------------------------------------------------
# FRQI state preparation
# ---------------------------------------------------------------------------

def apply_controlled_frqi_chunk_preparation(
    circuit: QuantumCircuit,
    chunk_index_reg: QuantumRegister,
    pixel_pos_reg: QuantumRegister,
    color_qubit,
    angles: np.ndarray,
    inverse: bool = False,
) -> None:
    """
    Prepare or unprepare an FRQI-style chunk state controlled by chunk index.

    Forward operation
    -----------------
    Starting from:

        |i>|0...0>|0>

    this prepares:

        |i> 1/sqrt(P) sum_t |t>(
            cos(theta_{i,t})|0> + sin(theta_{i,t})|1>
        )

    where P is the number of pixels in the chunk.

    Implementation
    --------------
    1. Apply H to the pixel-position register to create uniform |t>.
    2. For each chunk index i and pixel position t, apply:

            Ry(2 theta_{i,t})

       to the color qubit, controlled on:

            chunk_index == i
            pixel_position == t

    Inverse operation
    -----------------
    Applies the inverse rotations in reverse order, then removes the
    position superposition with H gates.

    Notes
    -----
    This does not precompute similarities. It only uses the image pixel values
    as rotation angles for state preparation.
    """
    num_chunks, pixels_per_chunk = angles.shape

    idx_bits = len(chunk_index_reg)
    pos_bits = len(pixel_pos_reg)

    index_controls = list(chunk_index_reg)
    position_controls = list(pixel_pos_reg)
    all_controls = index_controls + position_controls

    if not inverse:
        # Create uniform superposition over pixel positions.
        for q in pixel_pos_reg:
            circuit.h(q)

        # Apply controlled color rotations.
        for i in range(num_chunks):
            i_bits = int_to_little_endian_bits(i, idx_bits)

            for t in range(pixels_per_chunk):
                theta = float(angles[i, t])
                angle = 2.0 * theta

                if abs(angle) < 1e-12:
                    continue

                t_bits = int_to_little_endian_bits(t, pos_bits)
                basis_bits = i_bits + t_bits

                apply_controlled_ry_on_basis_state(
                    circuit=circuit,
                    controls=all_controls,
                    basis_bits_little_endian=basis_bits,
                    target=color_qubit,
                    angle=angle,
                )

    else:
        # Inverse controlled rotations, in reverse order.
        for i in reversed(range(num_chunks)):
            i_bits = int_to_little_endian_bits(i, idx_bits)

            for t in reversed(range(pixels_per_chunk)):
                theta = float(angles[i, t])
                angle = -2.0 * theta

                if abs(angle) < 1e-12:
                    continue

                t_bits = int_to_little_endian_bits(t, pos_bits)
                basis_bits = i_bits + t_bits

                apply_controlled_ry_on_basis_state(
                    circuit=circuit,
                    controls=all_controls,
                    basis_bits_little_endian=basis_bits,
                    target=color_qubit,
                    angle=angle,
                )

        # Remove uniform pixel-position superposition.
        for q in pixel_pos_reg:
            circuit.h(q)


# ---------------------------------------------------------------------------
# Coherent swap-test similarity phase oracle
# ---------------------------------------------------------------------------

def apply_swap_test_phase_oracle(
    circuit: QuantumCircuit,
    swap_ancilla,
    pos_i_reg: QuantumRegister,
    color_i,
    pos_j_reg: QuantumRegister,
    color_j,
) -> None:
    """
    Apply a coherent swap-test-based similarity phase.

    Swap test structure
    -------------------
    For states |C_i> and |C_j>:

        H on ancilla
        controlled-SWAP between |C_i> and |C_j>
        H on ancilla

    The ancilla has:

        Pr(0) = (1 + |<C_i|C_j>|^2) / 2

    This function applies a phase flip to the ancilla-|0> branch and then
    uncomputes the swap test.

    Effect
    ------
    This is a soft similarity oracle.

    If |C_i> == |C_j>, the ancilla would be |0> with probability 1, so the
    operation behaves like a clean -1 phase mark.

    If |C_i> and |C_j> are dissimilar, the phase effect is weaker.

    This is not a hard threshold predicate.
    """
    # Forward swap test.
    circuit.h(swap_ancilla)

    for a, b in zip(pos_i_reg, pos_j_reg):
        circuit.cswap(swap_ancilla, a, b)

    circuit.cswap(swap_ancilla, color_i, color_j)

    circuit.h(swap_ancilla)

    # Phase-flip the ancilla-|0> branch.
    apply_phase_flip_on_zero(circuit, swap_ancilla)

    # Uncompute swap test.
    circuit.h(swap_ancilla)

    circuit.cswap(swap_ancilla, color_i, color_j)

    for a, b in reversed(list(zip(pos_i_reg, pos_j_reg))):
        circuit.cswap(swap_ancilla, a, b)

    circuit.h(swap_ancilla)


def apply_self_pair_cancellation(
    circuit: QuantumCircuit,
    i_reg: QuantumRegister,
    j_reg: QuantumRegister,
    num_chunks: int,
) -> None:
    """
    Cancel trivial self-pairs i == j.

    The swap-test phase oracle strongly marks self-pairs because every chunk
    is identical to itself. We do not want those.

    For every index k, apply another -1 phase to |k>|k>. Since self-pairs
    already received a -1 phase, this second -1 cancels them:

        (-1) * (-1) = +1
    """
    idx_bits = len(i_reg)
    pair_controls = list(i_reg) + list(j_reg)

    for k in range(num_chunks):
        k_bits = int_to_little_endian_bits(k, idx_bits)
        basis_bits = k_bits + k_bits

        apply_phase_flip_on_basis_state(
            circuit=circuit,
            controls=pair_controls,
            basis_bits_little_endian=basis_bits,
        )


def apply_frqi_similarity_oracle(
    circuit: QuantumCircuit,
    i_reg: QuantumRegister,
    j_reg: QuantumRegister,
    pos_i_reg: QuantumRegister,
    pos_j_reg: QuantumRegister,
    color_i,
    color_j,
    swap_ancilla,
    angles: np.ndarray,
) -> None:
    """
    Full FRQI similarity oracle.

    This oracle does NOT precompute pairwise similarities.

    Steps
    -----
    1. Prepare |C_i> controlled by index i.
    2. Prepare |C_j> controlled by index j.
    3. Apply coherent swap-test phase oracle.
    4. Cancel trivial self-pairs i == j.
    5. Unprepare |C_j>.
    6. Unprepare |C_i>.

    At the end:
        - pos_i, pos_j, color_i, color_j, and swap_ancilla return to |0>.
        - only the pair-index registers keep the phase effect.
    """
    num_chunks = angles.shape[0]

    # Prepare FRQI states for the two indexed chunks.
    apply_controlled_frqi_chunk_preparation(
        circuit=circuit,
        chunk_index_reg=i_reg,
        pixel_pos_reg=pos_i_reg,
        color_qubit=color_i,
        angles=angles,
        inverse=False,
    )

    apply_controlled_frqi_chunk_preparation(
        circuit=circuit,
        chunk_index_reg=j_reg,
        pixel_pos_reg=pos_j_reg,
        color_qubit=color_j,
        angles=angles,
        inverse=False,
    )

    # Apply coherent similarity-dependent phase.
    apply_swap_test_phase_oracle(
        circuit=circuit,
        swap_ancilla=swap_ancilla,
        pos_i_reg=pos_i_reg,
        color_i=color_i,
        pos_j_reg=pos_j_reg,
        color_j=color_j,
    )

    # Remove self-pairs from the marked/amplified set.
    apply_self_pair_cancellation(
        circuit=circuit,
        i_reg=i_reg,
        j_reg=j_reg,
        num_chunks=num_chunks,
    )

    # Uncompute FRQI states to clean workspace.
    apply_controlled_frqi_chunk_preparation(
        circuit=circuit,
        chunk_index_reg=j_reg,
        pixel_pos_reg=pos_j_reg,
        color_qubit=color_j,
        angles=angles,
        inverse=True,
    )

    apply_controlled_frqi_chunk_preparation(
        circuit=circuit,
        chunk_index_reg=i_reg,
        pixel_pos_reg=pos_i_reg,
        color_qubit=color_i,
        angles=angles,
        inverse=True,
    )


# ---------------------------------------------------------------------------
# Grover diffusion
# ---------------------------------------------------------------------------

def apply_grover_diffusion(
    circuit: QuantumCircuit,
    search_qubits: List,
) -> None:
    """
    Apply Grover diffusion over the search register.

    Search register here is:

        |i>|j>
    """
    for q in search_qubits:
        circuit.h(q)

    for q in search_qubits:
        circuit.x(q)

    one_pattern = [1] * len(search_qubits)
    apply_phase_flip_on_basis_state(circuit, search_qubits, one_pattern)

    for q in search_qubits:
        circuit.x(q)

    for q in search_qubits:
        circuit.h(q)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_frqi_quantum_similarity_grover_circuit(
    angles: np.ndarray,
    grover_iterations: int,
) -> Tuple[QuantumCircuit, int]:
    """
    Build the full circuit.

    Qubit registers
    ---------------
    i_reg:
        Index of first chunk.

    j_reg:
        Index of second chunk.

    pos_i_reg, color_i:
        FRQI state register for chunk i.

    pos_j_reg, color_j:
        FRQI state register for chunk j.

    swap_ancilla:
        Ancilla for coherent swap-test phase oracle.

    c_reg:
        Classical bits for measured i and j.

    Qubit count
    -----------
    Let:
        m = number of chunks
        P = pixels per chunk

    Then:
        index qubits: 2 log2(m)
        FRQI state qubits: 2(log2(P) + 1)
        swap ancilla: 1

    Total:
        2 log2(m) + 2 log2(P) + 3

    This is much smaller than storing full chunks bitwise.
    """
    num_chunks, pixels_per_chunk = angles.shape

    if not is_power_of_two(num_chunks):
        raise ValueError(f"num_chunks must be a power of two. Got {num_chunks}.")

    if not is_power_of_two(pixels_per_chunk):
        raise ValueError(
            f"pixels_per_chunk must be a power of two. Got {pixels_per_chunk}."
        )

    if grover_iterations < 0:
        raise ValueError("grover_iterations must be nonnegative.")

    idx_bits = int(math.log2(num_chunks))
    pos_bits = int(math.log2(pixels_per_chunk))

    i_reg = QuantumRegister(idx_bits, "i")
    j_reg = QuantumRegister(idx_bits, "j")

    pos_i_reg = QuantumRegister(pos_bits, "pos_i")
    pos_j_reg = QuantumRegister(pos_bits, "pos_j")

    color_i_reg = QuantumRegister(1, "color_i")
    color_j_reg = QuantumRegister(1, "color_j")

    swap_anc_reg = QuantumRegister(1, "swap_anc")

    c_reg = ClassicalRegister(2 * idx_bits, "c")

    circuit = QuantumCircuit(
        i_reg,
        j_reg,
        pos_i_reg,
        pos_j_reg,
        color_i_reg,
        color_j_reg,
        swap_anc_reg,
        c_reg,
    )

    search_qubits = list(i_reg) + list(j_reg)

    # Prepare uniform superposition over all ordered chunk pairs |i>|j>.
    for q in search_qubits:
        circuit.h(q)

    # Grover-style iterations.
    for _ in range(grover_iterations):
        apply_frqi_similarity_oracle(
            circuit=circuit,
            i_reg=i_reg,
            j_reg=j_reg,
            pos_i_reg=pos_i_reg,
            pos_j_reg=pos_j_reg,
            color_i=color_i_reg[0],
            color_j=color_j_reg[0],
            swap_ancilla=swap_anc_reg[0],
            angles=angles,
        )

        apply_grover_diffusion(
            circuit=circuit,
            search_qubits=search_qubits,
        )

    # Measure only the pair-index registers.
    for bit_pos in range(idx_bits):
        circuit.measure(i_reg[bit_pos], c_reg[bit_pos])
        circuit.measure(j_reg[bit_pos], c_reg[idx_bits + bit_pos])

    return circuit, idx_bits


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def frqi_overlap_squared_for_diagnostics(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
) -> float:
    """
    Classical diagnostic only.

    This is NOT used to build the oracle. It is used only after measurement
    to print the similarity of measured pairs.
    """
    theta_a = chunk_a.flatten(order="C") * (math.pi / 2.0)
    theta_b = chunk_b.flatten(order="C") * (math.pi / 2.0)

    inner = float(np.mean(np.cos(theta_a - theta_b)))
    return inner * inner


def parse_measured_pair(bitstring: str, idx_bits: int) -> Tuple[int, int]:
    """
    Parse Qiskit count key into measured pair (i, j).

    Qiskit prints classical bits with highest bit on the left, so we reverse.
    """
    raw = bitstring.replace(" ", "")
    classical_bits_le = list(reversed(raw))

    i_bits = [int(classical_bits_le[k]) for k in range(idx_bits)]
    j_bits = [
        int(classical_bits_le[idx_bits + k])
        for k in range(idx_bits)
    ]

    i = bits_to_int_little_endian(i_bits)
    j = bits_to_int_little_endian(j_bits)

    return i, j


def print_top_results(
    counts: Dict[str, int],
    idx_bits: int,
    chunks: List[np.ndarray],
    positions: List[Tuple[int, int]],
    max_rows: int = 10,
) -> Tuple[int, int]:
    """
    Print top measured pairs and return the most frequent non-self pair.

    Self-pairs (i == j) are not considered valid answers, even if measured
    frequently. This is necessary because the soft FRQI oracle does not create
    a strict Boolean exclusion of self-pairs.
    """
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    print("\nTop measured pairs")
    print("------------------")

    best_nonself_pair = None

    for rank, (bitstring, count) in enumerate(ranked[:max_rows], start=1):
        i, j = parse_measured_pair(bitstring, idx_bits)

        is_valid_index = (
            0 <= i < len(chunks)
            and 0 <= j < len(chunks)
        )

        is_self_pair = i == j

        pos_i = positions[i] if 0 <= i < len(positions) else None
        pos_j = positions[j] if 0 <= j < len(positions) else None

        if is_valid_index:
            sim = frqi_overlap_squared_for_diagnostics(chunks[i], chunks[j])
            sim_text = f"{sim:.6f}"
        else:
            sim_text = "N/A"

        if best_nonself_pair is None and is_valid_index and not is_self_pair:
            best_nonself_pair = (i, j)

        status = "REJECT self-pair" if is_self_pair else "candidate"

        print(
            f"count={count:5d} | "
            f"i={i:3d}, j={j:3d} | "
            f"pos_i={pos_i}, pos_j={pos_j} | "
            f"diagnostic FRQI overlap^2={sim_text} | "
            f"self_pair={is_self_pair} | "
            f"{status}"
        )

    if best_nonself_pair is None:
        raise RuntimeError(
            "No non-self pair was found in the measured results. "
            "Try increasing shots, changing iterations, or using a stronger oracle."
        )

    print(f"\nSelected best non-self pair: {best_nonself_pair}")

    return best_nonself_pair


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRQI-style quantum similarity Grover demo without precomputed similarities."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Square chunk size in pixels.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help=(
            "Number of Grover-style iterations. "
            "Default: 1. Tune manually because this soft oracle has no known marked count."
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Number of simulator shots. Default: 2048.",
    )

    parser.add_argument(
        "--draw",
        action="store_true",
        help="Print the circuit diagram. This can be very large.",
    )

    args = parser.parse_args()

    image = load_grayscale_image(args.image)

    chunks, positions, cropped_shape = split_into_nonoverlapping_chunks(
        image=image,
        chunk_size=args.chunk_size,
    )

    num_chunks = len(chunks)
    pixels_per_chunk = args.chunk_size * args.chunk_size

    if not is_power_of_two(num_chunks):
        raise ValueError(
            f"This demo requires a power-of-two number of chunks. Got {num_chunks}.\n"
            f"Image cropped shape: {cropped_shape}, chunk size: {args.chunk_size}."
        )

    if not is_power_of_two(pixels_per_chunk):
        raise ValueError(
            f"This demo requires chunk_size^2 to be a power of two. "
            f"Got chunk_size^2 = {pixels_per_chunk}."
        )

    angles = chunks_to_angles(chunks)

    circuit, idx_bits = build_frqi_quantum_similarity_grover_circuit(
        angles=angles,
        grover_iterations=args.iterations,
    )

    print("Problem summary")
    print("---------------")
    print(f"Input image:                    {args.image}")
    print(f"Cropped image shape used:       {cropped_shape}")
    print(f"Chunk size:                     {args.chunk_size} x {args.chunk_size}")
    print(f"Number of chunks m:             {num_chunks}")
    print(f"Pixels per chunk P:             {pixels_per_chunk}")
    print(f"Index qubits per register:      {idx_bits}")
    print(f"Pixel-position qubits:          {int(math.log2(pixels_per_chunk))}")
    print(f"Grover-style iterations:        {args.iterations}")

    print("\nCircuit summary")
    print("---------------")
    print(f"Total qubits:                   {circuit.num_qubits}")
    print(f"Total classical bits:           {circuit.num_clbits}")
    print(f"Circuit depth before transpile: {circuit.depth()}")

    if args.draw:
        print("\nCircuit diagram")
        print("---------------")
        print(circuit.draw(output="text"))

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)

    print(f"Circuit depth after transpile:  {transpiled.depth()}")

    result = backend.run(transpiled, shots=args.shots).result()
    counts = result.get_counts()

    best_pair = print_top_results(
        counts=counts,
        idx_bits=idx_bits,
        chunks=chunks,
        positions=positions,
        max_rows=10,
    )

    highlight_matching_chunks(
            image_path=args.image,
            positions=positions,
            pair=best_pair,
            chunk_size=args.chunk_size,
            output_path="highlighted-frqi.png"
        )


if __name__ == "__main__":
    main()