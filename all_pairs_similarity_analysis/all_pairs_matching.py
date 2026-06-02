#!/usr/bin/env python3
"""
Quantum duplicate image-chunk finder using Grover search.

Goal
----
Given an image and a chunk size, split the image into non-overlapping chunks.
Find a pair of distinct chunks (i, j), i != j, such that chunk_i == chunk_j.

This script implements the quantum search structure:

    1. Prepare an equal superposition over chunk-index pairs |i>|j>.
    2. Reversibly load chunk_i and chunk_j into data registers.
    3. Reversibly compare the two chunk registers.
    4. Phase-flip states where chunk_i == chunk_j and i != j.
    5. Uncompute temporary registers.
    6. Apply Grover diffusion over the pair registers.
    7. Measure a likely duplicate pair.

Important
---------
This is a pedagogical simulator demo, not a scalable implementation.

Use tiny images and tiny chunks. The number of qubits is roughly:

    2 * log2(number_of_chunks) + 3 * bits_per_chunk

because we need:
    - i register
    - j register
    - chunk_i register
    - chunk_j register
    - diff register

Example
-------
Create a tiny test image, then run:

    python quantum_duplicate_chunks.py --image test.png --chunk-size 2 --bits-per-pixel 1 --shots 2048

Dependencies
------------
    pip install qiskit qiskit-aer pillow numpy
"""

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from .utils import int_to_little_endian_bits, apply_x_for_zero_controls
from .qrom_black_box import apply_qrom_lookup
from .visualization import highlight_matching_chunks

# ---------------------------------------------------------------------------
# Basic utility functions
# ---------------------------------------------------------------------------

def is_power_of_two(x: int) -> bool:
    """Return True iff x is a positive power of two."""
    return x > 0 and (x & (x - 1)) == 0


def bits_to_int_little_endian(bits: List[int]) -> int:
    """Convert little-endian bits back to an integer."""
    out = 0
    for k, bit in enumerate(bits):
        out |= (int(bit) << k)
    return out


# ---------------------------------------------------------------------------
# Image ingestion and chunk encoding
# ---------------------------------------------------------------------------

def load_image_as_quantized_gray(
    image_path: str,
    bits_per_pixel: int,
) -> np.ndarray:
    """
    Load an image, convert to grayscale, and quantize each pixel.

    If bits_per_pixel = 1:
        pixel values become 0 or 1 using threshold 128.

    If bits_per_pixel = q:
        pixel values become integers in {0, ..., 2^q - 1}.

    Returns
    -------
    quantized : np.ndarray of shape (height, width), dtype uint8
    """
    if bits_per_pixel < 1 or bits_per_pixel > 8:
        raise ValueError("bits_per_pixel must be between 1 and 8.")

    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    levels = 2 ** bits_per_pixel

    if bits_per_pixel == 1:
        quantized = (arr >= 128).astype(np.uint8)
    else:
        # Map 0..255 into 0..levels-1.
        quantized = np.floor(arr.astype(np.float64) * levels / 256.0).astype(np.uint8)
        quantized = np.clip(quantized, 0, levels - 1).astype(np.uint8)

    return quantized


def split_into_nonoverlapping_chunks(
    quantized: np.ndarray,
    chunk_size: int,
    bits_per_pixel: int,
) -> Tuple[List[List[int]], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Split the quantized image into non-overlapping square chunks.

    Each chunk is encoded as a little-endian bitstring.

    For each pixel in row-major order:
        append its bits in little-endian order.

    Example for bits_per_pixel=1:
        2x2 chunk:
            [[1, 0],
             [1, 1]]
        becomes:
            [1, 0, 1, 1]

    Example for bits_per_pixel=2:
        pixel value 3 = binary 11 -> bits [1, 1]
        pixel value 2 = binary 10 -> bits [0, 1]
        etc.

    Returns
    -------
    chunks:
        List of chunk bitstrings.
    positions:
        List of top-left pixel coordinates (row, col) for each chunk.
    cropped_shape:
        Shape of the cropped image region actually used.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    height, width = quantized.shape

    usable_height = (height // chunk_size) * chunk_size
    usable_width = (width // chunk_size) * chunk_size

    if usable_height == 0 or usable_width == 0:
        raise ValueError("Image is smaller than one chunk.")

    cropped = quantized[:usable_height, :usable_width]

    chunks: List[List[int]] = []
    positions: List[Tuple[int, int]] = []

    for row in range(0, usable_height, chunk_size):
        for col in range(0, usable_width, chunk_size):
            patch = cropped[row:row + chunk_size, col:col + chunk_size]

            bits: List[int] = []
            for pixel_value in patch.flatten(order="C"):
                bits.extend(int_to_little_endian_bits(int(pixel_value), bits_per_pixel))

            chunks.append(bits)
            positions.append((row, col))

    return chunks, positions, (usable_height, usable_width)


def find_classical_duplicate_pairs(chunks: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Classically find all ordered duplicate pairs (i, j), i != j.

    This is only used:
        1. to choose the Grover iteration count for the demo;
        2. to verify the measured result.

    The quantum circuit itself still performs chunk loading and equality testing.
    """
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    for idx, chunk in enumerate(chunks):
        groups[tuple(chunk)].append(idx)

    ordered_pairs: List[Tuple[int, int]] = []

    for indices in groups.values():
        if len(indices) >= 2:
            for i in indices:
                for j in indices:
                    if i != j:
                        ordered_pairs.append((i, j))

    return ordered_pairs



def apply_phase_flip_on_basis_state(
    circuit: QuantumCircuit,
    controls: List,
    basis_bits_little_endian: List[int],
) -> None:
    """
    Apply a -1 phase to exactly one computational basis state.

    That is, apply:
        |basis_bits> -> -|basis_bits>

    and leave all other basis states unchanged.

    Implementation:
        1. X controls that should be 0.
        2. Apply multi-controlled Z.
        3. Undo the X gates.
    """
    if len(controls) != len(basis_bits_little_endian):
        raise ValueError("controls and basis_bits_little_endian must have same length.")

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)

    n = len(controls)

    if n == 0:
        # Global phase. Rarely useful, but mathematically defined.
        circuit.global_phase += math.pi
    elif n == 1:
        circuit.z(controls[0])
    else:
        # Multi-controlled Z using H-MCX-H on the last qubit.
        target = controls[-1]
        ctrl = controls[:-1]
        circuit.h(target)
        circuit.mcx(ctrl, target)
        circuit.h(target)

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)


def apply_phase_flip_when_all_qubits_zero(
    circuit: QuantumCircuit,
    qubits: List,
) -> None:
    """
    Apply a -1 phase iff all given qubits are |0>.

    Used to phase-mark equality after diff = chunk_i XOR chunk_j.

    Equality condition:
        diff == 000...0
    """
    zero_pattern = [0] * len(qubits)
    apply_phase_flip_on_basis_state(circuit, qubits, zero_pattern)


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------

def apply_duplicate_pair_oracle(
    circuit: QuantumCircuit,
    i_reg: QuantumRegister,
    j_reg: QuantumRegister,
    chunk_i_reg: QuantumRegister,
    chunk_j_reg: QuantumRegister,
    diff_reg: QuantumRegister,
    chunks: List[List[int]],
) -> None:
    """
    Phase-flip pair states |i>|j> where chunk_i == chunk_j and i != j.

    The oracle is implemented as:

        1. Load chunk_i:
             |i>|0> -> |i>|chunk_i>

        2. Load chunk_j:
             |j>|0> -> |j>|chunk_j>

        3. Compute bitwise difference:
             diff = chunk_i XOR chunk_j

        4. Phase-flip if diff == 0.
             This marks all equal pairs, including trivial self-pairs i == j.

        5. Cancel the trivial self-pairs i == j by applying a second phase flip
           to states |i>|i>. Since (-1) * (-1) = +1, self-pairs are unmarked.

        6. Uncompute diff.

        7. Unload chunk_j and chunk_i by applying the same qROM lookups again.

    At the end, the data and diff registers are returned to |0...0>.
    Only the phase of the pair-index registers has changed.
    """
    num_chunks = len(chunks)
    idx_bits = len(i_reg)

    # -----------------------------------------------------------------------
    # 1. Load chunk_i and chunk_j into their data registers.
    # -----------------------------------------------------------------------
    apply_qrom_lookup(circuit, i_reg, chunk_i_reg, chunks)
    apply_qrom_lookup(circuit, j_reg, chunk_j_reg, chunks)

    # -----------------------------------------------------------------------
    # 2. Compute diff = chunk_i XOR chunk_j.
    # -----------------------------------------------------------------------
    for bit_pos in range(len(chunk_i_reg)):
        circuit.cx(chunk_i_reg[bit_pos], diff_reg[bit_pos])
        circuit.cx(chunk_j_reg[bit_pos], diff_reg[bit_pos])

    # -----------------------------------------------------------------------
    # 3. Phase-flip all states where diff == 0, i.e., chunks are equal.
    # -----------------------------------------------------------------------
    apply_phase_flip_when_all_qubits_zero(circuit, list(diff_reg))

    # -----------------------------------------------------------------------
    # 4. Cancel trivial self-pairs i == j.
    #
    # Self-pairs were marked above because every chunk equals itself.
    # We apply another phase flip to |i>|i>, which removes them.
    # -----------------------------------------------------------------------
    pair_controls = list(i_reg) + list(j_reg)

    # TODO: What??
    for idx in range(num_chunks):
        idx_bits_le = int_to_little_endian_bits(idx, idx_bits)
        basis_bits = idx_bits_le + idx_bits_le
        apply_phase_flip_on_basis_state(circuit, pair_controls, basis_bits)

    # TODO: Why??
    # -----------------------------------------------------------------------
    # 5. Uncompute diff.
    #
    # Since XOR is its own inverse, repeat the same CNOTs in reverse order.
    # -----------------------------------------------------------------------
    for bit_pos in reversed(range(len(chunk_i_reg))):
        circuit.cx(chunk_j_reg[bit_pos], diff_reg[bit_pos])
        circuit.cx(chunk_i_reg[bit_pos], diff_reg[bit_pos])

    # TODO: Why??
    # -----------------------------------------------------------------------
    # 6. Unload chunk_j and chunk_i.
    #
    # qROM lookup is XOR-based, so applying it again reverses it.
    # -----------------------------------------------------------------------
    apply_qrom_lookup(circuit, j_reg, chunk_j_reg, chunks)
    apply_qrom_lookup(circuit, i_reg, chunk_i_reg, chunks)


# ---------------------------------------------------------------------------
# Grover diffusion
# ---------------------------------------------------------------------------

def apply_grover_diffusion(
    circuit: QuantumCircuit,
    search_qubits: List,
) -> None:
    """
    Apply the Grover diffusion operator over the search space.

    For search register size s, this implements reflection about the uniform
    superposition:

        2|s><s| - I

    Standard circuit:
        H on all search qubits
        X on all search qubits
        phase flip |00...0> or equivalently MCZ on |11...1>
        X on all search qubits
        H on all search qubits
    """
    for q in search_qubits:
        circuit.h(q)

    for q in search_qubits:
        circuit.x(q)

    # Phase-flip |11...1>.
    one_pattern = [1] * len(search_qubits)
    apply_phase_flip_on_basis_state(circuit, search_qubits, one_pattern)

    for q in search_qubits:
        circuit.x(q)

    for q in search_qubits:
        circuit.h(q)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_duplicate_chunk_grover_circuit(
    chunks: List[List[int]],
    grover_iterations: int,
) -> Tuple[QuantumCircuit, int]:
    """
    Build the full Grover circuit.

    Returns
    -------
    circuit:
        Qiskit QuantumCircuit ready to simulate.
    idx_bits:
        Number of qubits in each chunk-index register.
    """
    num_chunks = len(chunks)

    if not is_power_of_two(num_chunks):
        raise ValueError(
            f"Number of chunks must be a power of two for this simple demo. "
            f"Got {num_chunks}."
        )

    if num_chunks < 2:
        raise ValueError("Need at least two chunks.")

    chunk_bits = len(chunks[0])

    for chunk in chunks:
        if len(chunk) != chunk_bits:
            raise ValueError("All chunks must have the same bit length.")

    idx_bits = int(math.log2(num_chunks))

    # -----------------------------------------------------------------------
    # Registers:
    #
    # i_reg       : first chunk index
    # j_reg       : second chunk index
    # chunk_i_reg : loaded chunk_i data
    # chunk_j_reg : loaded chunk_j data
    # diff_reg    : chunk_i XOR chunk_j
    # c_reg       : classical measurement bits for i and j only
    # -----------------------------------------------------------------------
    i_reg = QuantumRegister(idx_bits, "i")
    j_reg = QuantumRegister(idx_bits, "j")
    chunk_i_reg = QuantumRegister(chunk_bits, "chunk_i")
    chunk_j_reg = QuantumRegister(chunk_bits, "chunk_j")
    diff_reg = QuantumRegister(chunk_bits, "diff")
    c_reg = ClassicalRegister(2 * idx_bits, "c")

    circuit = QuantumCircuit(i_reg, j_reg, chunk_i_reg, chunk_j_reg, diff_reg, c_reg)

    # -----------------------------------------------------------------------
    # Prepare equal superposition over all ordered pairs |i>|j>.
    # -----------------------------------------------------------------------
    for q in list(i_reg) + list(j_reg):
        circuit.h(q)

    # -----------------------------------------------------------------------
    # Grover iterations:
    #     oracle -> diffusion
    # -----------------------------------------------------------------------
    search_qubits = list(i_reg) + list(j_reg)

    for _ in range(grover_iterations):
        apply_duplicate_pair_oracle(
            circuit=circuit,
            i_reg=i_reg,
            j_reg=j_reg,
            chunk_i_reg=chunk_i_reg,
            chunk_j_reg=chunk_j_reg,
            diff_reg=diff_reg,
            chunks=chunks,
        )

        apply_grover_diffusion(circuit, search_qubits)

    # -----------------------------------------------------------------------
    # Measure only the pair-index registers.
    #
    # Classical bit layout:
    #     c[0 : idx_bits]              stores i, little-endian
    #     c[idx_bits : 2*idx_bits]     stores j, little-endian
    # -----------------------------------------------------------------------
    for bit_pos in range(idx_bits):
        circuit.measure(i_reg[bit_pos], c_reg[bit_pos])
        circuit.measure(j_reg[bit_pos], c_reg[idx_bits + bit_pos])

    return circuit, idx_bits


# ---------------------------------------------------------------------------
# Measurement parsing and reporting
# ---------------------------------------------------------------------------

def parse_measured_pair(bitstring: str, idx_bits: int) -> Tuple[int, int]:
    """
    Parse a Qiskit count key into measured indices (i, j).

    Qiskit count keys are printed with the highest classical bit on the left.
    We reverse the string to recover c[0], c[1], ..., c[n-1].

    Our measurement layout is:
        c[0 : idx_bits]              -> i, little-endian
        c[idx_bits : 2*idx_bits]     -> j, little-endian
    """
    raw = bitstring.replace(" ", "")
    classical_bits_little_endian = list(reversed(raw))

    i_bits = [int(classical_bits_little_endian[k]) for k in range(idx_bits)]
    j_bits = [
        int(classical_bits_little_endian[idx_bits + k])
        for k in range(idx_bits)
    ]

    i = bits_to_int_little_endian(i_bits)
    j = bits_to_int_little_endian(j_bits)

    return i, j


def choose_grover_iterations(
    num_chunks: int,
    num_good_ordered_pairs: int,
) -> int:
    """
    Choose the usual near-optimal number of Grover iterations.

    Search space size:
        S = num_chunks^2

    Number of good states:
        k = num_good_ordered_pairs

    Approximate iteration count:
        floor(pi/4 * sqrt(S/k))

    If k == 0, there is no marked state. Return 0.
    """
    if num_good_ordered_pairs <= 0:
        return 0

    search_space_size = num_chunks * num_chunks

    return max(1, int(math.floor((math.pi / 4.0) * math.sqrt(search_space_size / num_good_ordered_pairs))))


def print_top_results(
    counts: Dict[str, int],
    idx_bits: int,
    chunks: List[List[int]],
    positions: List[Tuple[int, int]],
    max_rows: int = 10,
) -> None:
    """Print the most common measured pairs and whether they are true duplicates."""
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    print("\nTop measured pairs:")
    print("-------------------")

    for bitstring, count in ranked[:max_rows]:
        i, j = parse_measured_pair(bitstring, idx_bits)

        if i < len(chunks) and j < len(chunks):
            equal = chunks[i] == chunks[j]
            nontrivial = i != j
            valid_duplicate = equal and nontrivial
            pos_i = positions[i]
            pos_j = positions[j]
        else:
            equal = False
            nontrivial = False
            valid_duplicate = False
            pos_i = None
            pos_j = None

        print(
            f"count={count:5d} | "
            f"i={i:3d}, j={j:3d} | "
            f"pos_i={pos_i}, pos_j={pos_j} | "
            f"equal={equal} | i!=j={nontrivial} | "
            f"duplicate={valid_duplicate}"
        )


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find duplicate image chunks using Grover search in Qiskit."
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
        help="Square chunk size in pixels, e.g. 2 for 2x2 chunks.",
    )

    parser.add_argument(
        "--bits-per-pixel",
        type=int,
        default=1,
        help="Quantized grayscale bits per pixel. Use 1 for tiny demos. Default: 1.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Number of simulator shots. Default: 2048.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Override Grover iteration count. "
            "If omitted, the script uses the classically known number of duplicate pairs "
            "to choose a demo-friendly value."
        ),
    )

    parser.add_argument(
        "--draw",
        action="store_true",
        help="Print the circuit text diagram. Can be huge.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load and quantize the image.
    # -----------------------------------------------------------------------
    quantized = load_image_as_quantized_gray(
        image_path=args.image,
        bits_per_pixel=args.bits_per_pixel,
    )

    # -----------------------------------------------------------------------
    # Split the image into non-overlapping chunks and encode each as bits.
    # -----------------------------------------------------------------------
    chunks, positions, cropped_shape = split_into_nonoverlapping_chunks(
        quantized=quantized,
        chunk_size=args.chunk_size,
        bits_per_pixel=args.bits_per_pixel,
    )

    num_chunks = len(chunks)
    chunk_bits = len(chunks[0])

    if not is_power_of_two(num_chunks):
        raise ValueError(
            f"This simple demo requires the number of chunks to be a power of two.\n"
            f"Got {num_chunks} chunks from cropped image shape {cropped_shape} "
            f"with chunk size {args.chunk_size}.\n"
            f"Try an image/chunk-size combination giving 2, 4, 8, 16, ... chunks."
        )

    # -----------------------------------------------------------------------
    # Classical duplicate count is used only for choosing Grover iterations
    # and verifying results.
    # -----------------------------------------------------------------------
    duplicate_pairs = find_classical_duplicate_pairs(chunks)
    num_good = len(duplicate_pairs)

    if args.iterations is None:
        grover_iterations = choose_grover_iterations(num_chunks, num_good)
    else:
        grover_iterations = args.iterations

    # -----------------------------------------------------------------------
    # Print problem summary.
    # -----------------------------------------------------------------------
    print("Problem summary")
    print("---------------")
    print(f"Input image:                {args.image}")
    print(f"Cropped image shape used:   {cropped_shape}")
    print(f"Chunk size:                 {args.chunk_size} x {args.chunk_size}")
    print(f"Bits per pixel:             {args.bits_per_pixel}")
    print(f"Number of chunks m:         {num_chunks}")
    print(f"Bits per chunk b:           {chunk_bits}")
    print(f"Search space size m^2:      {num_chunks * num_chunks}")
    print(f"Classical good ordered pairs, excluding i=j: {num_good}")
    print(f"Grover iterations:          {grover_iterations}")

    if num_good > 0:
        print("\nClassically known duplicate ordered pairs, for verification:")
        for pair in duplicate_pairs[:20]:
            i, j = pair
            print(f"  ({i}, {j}) at positions {positions[i]} and {positions[j]}")
        if len(duplicate_pairs) > 20:
            print(f"  ... and {len(duplicate_pairs) - 20} more")
    else:
        print(
            "\nNo duplicate chunks were found classically. "
            "Grover has no marked state in this case, so measurement will be uninformative."
        )

    # -----------------------------------------------------------------------
    # Build the quantum circuit.
    # -----------------------------------------------------------------------
    circuit, idx_bits = build_duplicate_chunk_grover_circuit(
        chunks=chunks,
        grover_iterations=grover_iterations,
    )

    print("\nCircuit summary")
    print("---------------")
    print(f"Total qubits:               {circuit.num_qubits}")
    print(f"Total classical bits:       {circuit.num_clbits}")
    print(f"Circuit depth before transpile: {circuit.depth()}")

    if args.draw:
        print("\nCircuit diagram")
        print("---------------")
        print(circuit.draw(output="text"))

    # -----------------------------------------------------------------------
    # Simulate the circuit.
    # -----------------------------------------------------------------------
    backend = AerSimulator()
    transpiled = transpile(circuit, backend)

    print(f"Circuit depth after transpile:  {transpiled.depth()}")

    result = backend.run(transpiled, shots=args.shots).result()
    counts = result.get_counts()

    # -----------------------------------------------------------------------
    # Display and verify the top measured pairs.
    # -----------------------------------------------------------------------
    print_top_results(
        counts=counts,
        idx_bits=idx_bits,
        chunks=chunks,
        positions=positions,
        max_rows=10,
    )

    top_bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    top_pair = parse_measured_pair(top_bitstring, idx_bits)

    highlight_matching_chunks(
        image_path=args.image,
        positions=positions,
        pair=top_pair,
        chunk_size=args.chunk_size,
        output_path="highlighted_match.png",
        show=True,
    )


if __name__ == "__main__":
    main()