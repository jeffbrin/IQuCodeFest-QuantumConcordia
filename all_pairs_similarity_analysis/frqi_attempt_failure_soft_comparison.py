#!/usr/bin/env python3
"""
FRQI-style Grover search for visually similar image chunks.

Goal
----
Given an image and a chunk size, split the image into non-overlapping chunks.
Find a pair of distinct chunks (i, j), i != j, whose FRQI-style similarity is
above a chosen threshold.

This version is designed for a small Qiskit simulator demo.

Key difference from exact-equality version
------------------------------------------
The old exact-equality circuit needed large chunk registers:

    chunk_i register
    chunk_j register
    diff register

For n x n binary chunks, that cost roughly:

    3 * n^2 qubits

This FRQI-style version avoids storing raw chunks in quantum registers.
Instead, each chunk is interpreted as an FRQI-style state:

    |C_i> = 1/sqrt(P) sum_t |t> (
                cos(theta_{i,t}) |0> + sin(theta_{i,t}) |1>
            )

where:
    P = number of pixels in a chunk
    theta_{i,t} encodes pixel intensity

The similarity between two chunks is:

    overlap(i, j) = |<C_i | C_j>|^2

Pairs whose overlap exceeds a threshold are marked by the Grover oracle.

Important honesty note
----------------------
For this demo, the FRQI overlaps are computed classically when constructing
the oracle. The quantum circuit then performs Grover search over the pair
indices.

A fully coherent implementation would prepare |C_i> and |C_j>, estimate their
overlap using a swap test or amplitude estimation, compare to a threshold, flip
a phase, and uncompute. That is much more complex and much deeper.

Example
-------
    python frqi_grover_similarity.py \\
        --image test.png \\
        --chunk-size 16 \\
        --threshold 0.98 \\
        --shots 2048
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from visualization import highlight_matching_chunks


# ---------------------------------------------------------------------------
# Basic bit utilities
# ---------------------------------------------------------------------------

def is_power_of_two(x: int) -> bool:
    """Return True iff x is a positive power of two."""
    return x > 0 and (x & (x - 1)) == 0


def int_to_little_endian_bits(value: int, width: int) -> List[int]:
    """
    Convert integer to little-endian bit list.

    Example:
        value = 6 = binary 110
        width = 3
        returns [0, 1, 1]
    """
    return [(value >> k) & 1 for k in range(width)]


def bits_to_int_little_endian(bits: List[int]) -> int:
    """Convert little-endian bit list back to integer."""
    value = 0
    for k, bit in enumerate(bits):
        value |= int(bit) << k
    return value


# ---------------------------------------------------------------------------
# Image loading and chunking
# ---------------------------------------------------------------------------

def load_grayscale_image(image_path: str) -> np.ndarray:
    """
    Load an image as grayscale intensities normalized to [0, 1].

    Returns
    -------
    arr:
        Float array of shape (height, width), values in [0, 1].
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def split_into_nonoverlapping_chunks(
    image: np.ndarray,
    chunk_size: int,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Split image into non-overlapping square chunks.

    Parameters
    ----------
    image:
        Grayscale image as a float array in [0, 1].

    chunk_size:
        Width and height of each square chunk.

    Returns
    -------
    chunks:
        List of chunks, each shape (chunk_size, chunk_size).

    positions:
        positions[i] = (row, col), the top-left pixel of chunk i.

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
            chunk = cropped[row:row + chunk_size, col:col + chunk_size]
            chunks.append(chunk.copy())
            positions.append((row, col))

    return chunks, positions, (usable_height, usable_width)


# ---------------------------------------------------------------------------
# FRQI-style similarity
# ---------------------------------------------------------------------------

def chunk_to_frqi_angles(chunk: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale chunk into FRQI-style intensity angles.

    Pixel values are assumed to be in [0, 1].

    We encode intensity x as:

        theta = x * pi / 2

    so:
        black x=0   -> theta=0
        white x=1   -> theta=pi/2

    In FRQI, each pixel contributes a color qubit state:

        cos(theta)|0> + sin(theta)|1>
    """
    return chunk.flatten(order="C") * (math.pi / 2.0)


def frqi_overlap_squared(chunk_a: np.ndarray, chunk_b: np.ndarray) -> float:
    """
    Compute FRQI-style squared overlap between two chunks.

    For two FRQI-style chunk states:

        |C_a> = 1/sqrt(P) sum_t |t>(
                    cos(theta_a[t])|0> + sin(theta_a[t])|1>
                )

        |C_b> = 1/sqrt(P) sum_t |t>(
                    cos(theta_b[t])|0> + sin(theta_b[t])|1>
                )

    Their inner product is:

        <C_a|C_b> = (1/P) sum_t cos(theta_a[t] - theta_b[t])

    because:

        cos(a)cos(b) + sin(a)sin(b) = cos(a - b)

    The returned similarity is:

        |<C_a|C_b>|^2

    Values close to 1 mean very similar.
    Values close to 0 mean very different.
    """
    if chunk_a.shape != chunk_b.shape:
        raise ValueError("Chunks must have the same shape.")

    theta_a = chunk_to_frqi_angles(chunk_a)
    theta_b = chunk_to_frqi_angles(chunk_b)

    inner_product = float(np.mean(np.cos(theta_a - theta_b)))
    return inner_product * inner_product

def is_mostly_white_chunk(
    chunk: np.ndarray,
    white_threshold: float = 0.95,
    white_pixel_cutoff: float = 0.95,
) -> bool:
    """
    Return True if a chunk is mostly white.

    Parameters
    ----------
    chunk:
        Grayscale chunk with values in [0, 1].

    white_threshold:
        Fraction of pixels that must be white-ish for the chunk to be removed.

    white_pixel_cutoff:
        Pixel intensity above which a pixel is considered white.

    Example
    -------
    If white_threshold=0.95 and white_pixel_cutoff=0.95,
    then a chunk is removed if at least 95% of its pixels have intensity >= 0.95.
    """
    white_fraction = np.mean(chunk >= white_pixel_cutoff)
    return white_fraction >= white_threshold


def compute_marked_pairs_by_frqi_similarity(
    chunks: List[np.ndarray],
    threshold: float,
    white_threshold: float = 0.95,
    white_pixel_cutoff: float = 0.95,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Find ordered pairs (i, j), i != j, with FRQI overlap >= threshold.

    Mostly-white chunks are ignored so Grover does not amplify blank/background
    regions that are visually uninteresting.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1].")

    marked_pairs: List[Tuple[int, int]] = []
    similarities: Dict[Tuple[int, int], float] = {}

    num_chunks = len(chunks)

    # Precompute which chunks should be ignored.
    skip_chunk = [
        is_mostly_white_chunk(
            chunk,
            white_threshold=white_threshold,
            white_pixel_cutoff=white_pixel_cutoff,
        )
        for chunk in chunks
    ]

    ignored = sum(skip_chunk)
    print(f"Mostly-white chunks ignored: {ignored} / {num_chunks}")

    for i in range(num_chunks):
        if skip_chunk[i]:
            continue

        for j in range(num_chunks):
            if i == j:
                continue

            if skip_chunk[j]:
                continue

            sim = frqi_overlap_squared(chunks[i], chunks[j])
            similarities[(i, j)] = sim

            if sim >= threshold:
                marked_pairs.append((i, j))

    return marked_pairs, similarities

# ---------------------------------------------------------------------------
# Phase-oracle helpers
# ---------------------------------------------------------------------------

def apply_x_for_zero_controls(
    circuit: QuantumCircuit,
    controls: List,
    bit_pattern: List[int],
) -> None:
    """
    Qiskit's multi-controlled gates control on |1>.

    To control on a desired |0>, temporarily apply X before and after.
    """
    for qubit, bit in zip(controls, bit_pattern):
        if bit == 0:
            circuit.x(qubit)


def apply_phase_flip_on_basis_state(
    circuit: QuantumCircuit,
    controls: List,
    basis_bits_little_endian: List[int],
) -> None:
    """
    Apply a -1 phase to exactly one computational basis state.

    Example:
        controls = [q0, q1, q2]
        basis_bits_little_endian = [1, 0, 1]

    This phase-flips the state:

        |q2 q1 q0> = |1 0 1>

    while leaving all other basis states unchanged.
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

        # Multi-controlled Z using H-MCX-H on the last qubit.
        circuit.h(target)
        circuit.mcx(ctrl, target)
        circuit.h(target)

    apply_x_for_zero_controls(circuit, controls, basis_bits_little_endian)


# ---------------------------------------------------------------------------
# FRQI-similarity Grover oracle
# ---------------------------------------------------------------------------

def apply_frqi_similarity_oracle(
    circuit: QuantumCircuit,
    i_reg: QuantumRegister,
    j_reg: QuantumRegister,
    marked_pairs: List[Tuple[int, int]],
) -> None:
    """
    Phase-flip every pair (i, j) whose FRQI similarity is above threshold.

    This implements:

        |i>|j> -> -|i>|j>

    if (i, j) is in marked_pairs.

    This is a compiled phase oracle. For a small simulator demo, this is the
    cleanest way to show Grover amplification over FRQI-similar chunk pairs
    without allocating huge chunk registers.
    """
    idx_bits = len(i_reg)
    pair_controls = list(i_reg) + list(j_reg)

    for i, j in marked_pairs:
        i_bits = int_to_little_endian_bits(i, idx_bits)
        j_bits = int_to_little_endian_bits(j, idx_bits)
        basis_bits = i_bits + j_bits

        apply_phase_flip_on_basis_state(
            circuit=circuit,
            controls=pair_controls,
            basis_bits_little_endian=basis_bits,
        )


# ---------------------------------------------------------------------------
# Grover diffusion over pair-index registers
# ---------------------------------------------------------------------------

def apply_grover_diffusion(
    circuit: QuantumCircuit,
    search_qubits: List,
) -> None:
    """
    Apply Grover diffusion over the pair-index search space.

    Diffusion is reflection about the uniform superposition:

        2|s><s| - I

    where:

        |s> = uniform superposition over all |i>|j>.
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


def choose_grover_iterations(
    search_space_size: int,
    num_marked: int,
) -> int:
    """
    Choose a reasonable Grover iteration count.

    If:
        S = search_space_size
        k = num_marked

    Then sin^2(theta) = k / S.

    A common near-optimal iteration count is:

        r ~= floor(pi / (4 theta) - 1/2)

    If no states are marked, return 0.
    """
    if num_marked <= 0:
        return 0

    if num_marked >= search_space_size:
        return 0

    theta = math.asin(math.sqrt(num_marked / search_space_size))
    r = int(math.floor((math.pi / (4.0 * theta)) - 0.5))

    return max(1, r)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_frqi_similarity_grover_circuit(
    num_chunks: int,
    marked_pairs: List[Tuple[int, int]],
    grover_iterations: int,
) -> Tuple[QuantumCircuit, int]:
    """
    Build the Grover circuit over ordered chunk-index pairs.

    Registers
    ---------
    i_reg:
        First chunk index.

    j_reg:
        Second chunk index.

    c_reg:
        Classical output register storing measured i and j.

    Qubit count
    -----------
    This circuit uses only:

        2 * log2(num_chunks)

    qubits, because the FRQI similarity predicate has been compiled into
    the phase oracle.
    """
    if not is_power_of_two(num_chunks):
        raise ValueError(
            "This demo requires the number of chunks to be a power of two. "
            f"Got {num_chunks}."
        )

    idx_bits = int(math.log2(num_chunks))

    i_reg = QuantumRegister(idx_bits, "i")
    j_reg = QuantumRegister(idx_bits, "j")
    c_reg = ClassicalRegister(2 * idx_bits, "c")

    circuit = QuantumCircuit(i_reg, j_reg, c_reg)

    search_qubits = list(i_reg) + list(j_reg)

    # Prepare equal superposition over all ordered pairs |i>|j>.
    for q in search_qubits:
        circuit.h(q)

    # Repeated Grover iterations: oracle then diffusion.
    for _ in range(grover_iterations):
        apply_frqi_similarity_oracle(
            circuit=circuit,
            i_reg=i_reg,
            j_reg=j_reg,
            marked_pairs=marked_pairs,
        )

        apply_grover_diffusion(
            circuit=circuit,
            search_qubits=search_qubits,
        )

    # Measure i and j into classical bits.
    for bit_pos in range(idx_bits):
        circuit.measure(i_reg[bit_pos], c_reg[bit_pos])
        circuit.measure(j_reg[bit_pos], c_reg[idx_bits + bit_pos])

    return circuit, idx_bits


# ---------------------------------------------------------------------------
# Measurement parsing and reporting
# ---------------------------------------------------------------------------

def parse_measured_pair(bitstring: str, idx_bits: int) -> Tuple[int, int]:
    """
    Parse Qiskit count key into measured pair (i, j).

    Qiskit displays classical bits with the highest classical bit on the left.
    We reverse the string to get c[0], c[1], ..., c[n-1].
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
    positions: List[Tuple[int, int]],
    similarities: Dict[Tuple[int, int], float],
    threshold: float,
    max_rows: int = 10,
) -> Tuple[int, int]:
    """
    Print top measured pairs and return the most frequent pair.
    """
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    print("\nTop measured pairs")
    print("------------------")

    best_pair = None

    for rank, (bitstring, count) in enumerate(ranked[:max_rows], start=1):
        i, j = parse_measured_pair(bitstring, idx_bits)

        sim = similarities.get((i, j), None)
        marked = sim is not None and sim >= threshold and i != j

        pos_i = positions[i] if 0 <= i < len(positions) else None
        pos_j = positions[j] if 0 <= j < len(positions) else None

        if rank == 1:
            best_pair = (i, j)

        if sim is None:
            sim_text = "N/A"
        else:
            sim_text = f"{sim:.6f}"

        print(
            f"count={count:5d} | "
            f"i={i:3d}, j={j:3d} | "
            f"pos_i={pos_i}, pos_j={pos_j} | "
            f"FRQI overlap^2={sim_text} | "
            f"marked={marked}"
        )

    if best_pair is None:
        raise RuntimeError("No measurement results were returned.")

    return best_pair

# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRQI-style Grover search for similar image chunks."
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
        help="Square chunk size in pixels, e.g. 16 for 16x16 chunks.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help=(
            "FRQI overlap-squared threshold in [0, 1]. "
            "Higher means stricter similarity. Default: 0.98."
        ),
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
            "If omitted, a near-optimal value is chosen from the number of marked pairs."
        ),
    )

    parser.add_argument(
        "--draw",
        action="store_true",
        help="Print the circuit diagram. Can be large.",
    )

    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Save an image highlighting the most frequently measured pair.",
    )

    parser.add_argument(
        "--highlight-output",
        default="frqi_similarity_match.png",
        help="Output path for highlighted image.",
    )

    parser.add_argument(
        "--highlight-scale",
        type=int,
        default=4,
        help="Scale factor for highlighted output image. Default: 4.",
    )

    parser.add_argument(
    "--white-threshold",
    type=float,
    default=1,
    help=(
        "Remove chunks where this fraction of pixels are white-ish. "
        "Default: 0.95."
    ),
    )

    parser.add_argument(
        "--white-pixel-cutoff",
        type=float,
        default=1,
        help=(
            "Pixel intensity in [0, 1] above which a pixel is considered white. "
            "Default: 0.95."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load image and split into chunks.
    # -----------------------------------------------------------------------
    image = load_grayscale_image(args.image)

    chunks, positions, cropped_shape = split_into_nonoverlapping_chunks(
        image=image,
        chunk_size=args.chunk_size,
    )

    num_chunks = len(chunks)

    if not is_power_of_two(num_chunks):
        raise ValueError(
            f"This demo requires a power-of-two number of chunks.\n"
            f"Got {num_chunks} chunks from cropped image shape {cropped_shape} "
            f"with chunk size {args.chunk_size}.\n"
            f"Try choosing an image and chunk size giving 2, 4, 8, 16, ... chunks."
        )

    # -----------------------------------------------------------------------
    # 2. Compute FRQI-style similarity and marked states.
    # -----------------------------------------------------------------------
    marked_pairs, similarities = compute_marked_pairs_by_frqi_similarity(
        chunks=chunks,
        threshold=args.threshold,
        white_threshold=args.white_threshold,
        white_pixel_cutoff=args.white_pixel_cutoff,
    )

    search_space_size = num_chunks * num_chunks
    num_marked = len(marked_pairs)

    if args.iterations is None:
        grover_iterations = choose_grover_iterations(
            search_space_size=search_space_size,
            num_marked=num_marked,
        )
    else:
        grover_iterations = args.iterations

    # -----------------------------------------------------------------------
    # 3. Print problem summary.
    # -----------------------------------------------------------------------
    print("Problem summary")
    print("---------------")
    print(f"Input image:                     {args.image}")
    print(f"Cropped image shape used:        {cropped_shape}")
    print(f"Chunk size:                      {args.chunk_size} x {args.chunk_size}")
    print(f"Number of chunks m:              {num_chunks}")
    print(f"Search space size m^2:           {search_space_size}")
    print(f"FRQI overlap^2 threshold:        {args.threshold}")
    print(f"Marked ordered pairs:            {num_marked}")
    print(f"Grover iterations:               {grover_iterations}")

    if num_marked == 0:
        print(
            "\nNo pairs exceeded the threshold. "
            "Grover has no marked state, so measurement will be uninformative."
        )
    else:
        print("\nMarked pairs, first 20")
        print("----------------------")
        for i, j in marked_pairs[:20]:
            print(
                f"({i}, {j}) | "
                f"positions {positions[i]} and {positions[j]} | "
                f"overlap^2={similarities[(i, j)]:.6f}"
            )
        if num_marked > 20:
            print(f"... and {num_marked - 20} more")

    # -----------------------------------------------------------------------
    # 4. Build Grover circuit.
    # -----------------------------------------------------------------------
    circuit, idx_bits = build_frqi_similarity_grover_circuit(
        num_chunks=num_chunks,
        marked_pairs=marked_pairs,
        grover_iterations=grover_iterations,
    )

    print("\nCircuit summary")
    print("---------------")
    print(f"Index qubits per register:       {idx_bits}")
    print(f"Total qubits:                    {circuit.num_qubits}")
    print(f"Total classical bits:            {circuit.num_clbits}")
    print(f"Circuit depth before transpile:  {circuit.depth()}")

    if args.draw:
        print("\nCircuit diagram")
        print("---------------")
        print(circuit.draw(output="text"))

    # -----------------------------------------------------------------------
    # 5. Simulate.
    # -----------------------------------------------------------------------
    backend = AerSimulator()
    transpiled = transpile(circuit, backend)

    print(f"Circuit depth after transpile:   {transpiled.depth()}")

    result = backend.run(transpiled, shots=args.shots).result()
    counts = result.get_counts()

    # -----------------------------------------------------------------------
    # 6. Report results.
    # -----------------------------------------------------------------------
    best_pair = print_top_results(
        counts=counts,
        idx_bits=idx_bits,
        positions=positions,
        similarities=similarities,
        threshold=args.threshold,
        max_rows=10,
    )

    # -----------------------------------------------------------------------
    # 7. Optionally save highlighted image.
    # -----------------------------------------------------------------------
    if args.highlight:
        highlight_matching_chunks(
            image_path=args.image,
            positions=positions,
            pair=best_pair,
            chunk_size=args.chunk_size,
            output_path=args.highlight_output,
            scale=args.highlight_scale,
        )


if __name__ == "__main__":
    main()