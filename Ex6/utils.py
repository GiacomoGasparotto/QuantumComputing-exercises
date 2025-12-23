import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from time import time

from debugger import *

def separable_pure_state(subsys_dims, subsys_vectors=None):
    """
    Construct a separable pure state for a multipartite quantum system.

    Args:
        subsys_dims (list[int]):
            Dimensions of each subsystem.
        subsys_vectors (list[np.ndarray] or None):
            Optional list of state vectors for each subsystem.
            If None, random normalized complex vectors are generated.

    Returns:
        np.ndarray:
            Flattened array containing all subsystem state vectors
            concatenated in order.
    """
    
    if subsys_vectors is None: # Generate random vectors
        subsys_vectors = np.array([])

        for d in subsys_dims:
            # Generate a generic state
            Re = np.random.randn(d)
            Im = np.random.randn(d)
            psi = Re + 1j*Im

            # Normalize
            psi = psi/np.linalg.norm(psi)
            vector_checkpoints(0, psi, debug=True)

            # Fill the array
            subsys_vectors = np.append(subsys_vectors, psi)

        # Return the subsystem vectors
        return subsys_vectors
    
    else:
        # Return the subsystem vectors
        return np.concatenate(subsys_vectors)

def tensor_prod(subsys_dims, subsys_vectors=None):
    """
    Compute the tensor product of subsystem state vectors.

    Args:
        subsys_dims (list[int]):
            Dimensions of each subsystem.
        subsys_vectors (list[np.ndarray] or None):
            Optional list of subsystem state vectors.
            If None, random separable subsystem states are generated.

    Returns:
        np.ndarray:
            State vector of the composite system obtained via tensor product.
    """
    flat_vectors = separable_pure_state(subsys_dims, subsys_vectors)

    vectors_list = []
    start = 0

    # Compute the list of subsystem vectors
    for d in subsys_dims:
        end = start + d
        vec = flat_vectors[start:end]
        vectors_list.append(vec)
        start = end

    # Return the tensor product
    return reduce(np.kron, vectors_list)

def general_pure_state(subsys_dims, subsys_vectors=None):
    """
    Generate a general (possibly entangled) pure state of a composite system.

    Args:
        subsys_dims (list[int]):
            Dimensions of each subsystem.
        subsys_vectors (array-like or None):
            Optional full state vector of the composite system.
            If None, a random complex vector is generated.

    Returns:
        np.ndarray:
            Normalized state vector of the full Hilbert space.
    """

    # Compute the total dimension
    tot_dim = np.prod(subsys_dims)

    if subsys_vectors is None:

        # Generate a generic state 
        Re = np.random.randn(tot_dim)
        Im = np.random.randn(tot_dim)
        psi = Re + 1j*Im
    
    else:
        psi = np.array(subsys_vectors, dtype=complex)
        
        # Dimensions check
        if psi.size != tot_dim:
            raise ValueError(f"Warning: Input dimension {psi.size} differs from expected {tot_dim}")
    
    # Normalize the state
    psi = psi/np.linalg.norm(psi)
    vector_checkpoints(0, psi, debug=True)
    
    # Return the state vector
    return psi

def density_matrix(psi):
    """
    Construct the density matrix of a pure quantum state.

    Args:
        psi (np.ndarray):
            State vector of the system.

    Returns:
        np.ndarray:
            Density matrix rho = |psi><psi|.
    """

    # Return the density matrix
    return np.outer(psi, np.conj(psi))


def reduced_density_matrix(rho, dims, choose_subsystem=0):
    """
    Compute the reduced density matrix of a bipartite system
    by tracing out one subsystem.

    Args:
        rho (np.ndarray):
            Density matrix of the composite system.
        dims (list[int]):
            Dimensions of the two subsystems [dim_A, dim_B].
        choose_subsystem (int):
            Subsystem to keep:
            0 -> keep left subsystem (trace out right),
            1 -> keep right subsystem (trace out left).

    Returns:
        np.ndarray:
            Reduced density matrix of the selected subsystem.
    """

    # Check we have exactly two subsystems
    if len(dims) != 2:
        raise ValueError("ERROR. The number of subsystems must be equal to 2!")
    
    # Extract dimensions
    dim_left, dim_right = dims[0], dims[1]
    
    # Compute the reduced matrix
    if choose_subsystem==0:
        # Left subsytem
        rho_reduced = np.zeros((dim_left, dim_left), dtype=complex)
        for i in range(dim_left):
            for j in range(dim_left):
                sum = 0
                for k in range(dim_right):
                    sum += rho[i*dim_right+k, j*dim_right+k] 

                rho_reduced[i, j] = sum
    
    elif choose_subsystem==1:
        # Right subsystem
        rho_reduced = np.zeros((dim_right, dim_right), dtype=complex)
        for i in range(dim_right):
            for j in range(dim_right):
                sum = 0
                for k in range(dim_left):
                    sum += rho[k*dim_right+i, k*dim_right+j] 

                rho_reduced[i, j] = sum

    else:
        raise ValueError("system_to_keep must be 0 (Left) or 1 (Right)")

    # Return the reduced density matrix
    return rho_reduced

def density_trace(rho):
    """
    Compute the trace of a density matrix.

    Args:
        rho (np.ndarray):
            Density matrix.

    Returns:
        complex:
            Trace of rho.
    """

    # Return the trace
    return np.trace(rho)

def purity(rho):
    """
    Compute the purity of a quantum state.

    Args:
        rho (np.ndarray):
            Density matrix.

    Returns:
        float:
            Purity Tr(rho^2).
    """
    
    # Compute rho squared
    rho_square = rho@rho
    
    # Return the purity
    return density_trace(rho_square)

def plot_density_matrix(rho, rhoA, rhoB, savefig=None):
    """
    Plot the real parts of a bipartite density matrix and its reduced states.

    Args:
        rho (np.ndarray):
            Density matrix of the composite system.
        rhoA (np.ndarray):
            Reduced density matrix of subsystem A.
        rhoB (np.ndarray):
            Reduced density matrix of subsystem B.

    Returns:
        None
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))

    im = ax1.imshow(np.real(rho), cmap="coolwarm")
    ax1.set_title(r"$\rho_{AB}$")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(np.real(rhoA), cmap="coolwarm")
    ax2.set_title(r"$\rho_A = Tr_B(\rho_{AB})$")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(np.real(rhoB), cmap="coolwarm")
    ax3.set_title(r"$\rho_B = Tr_A(\rho_{AB})$")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Show colorbar
    plt.subplots_adjust(bottom=0.30)
    cax = fig.add_axes([0.25, 0.28, 0.53, 0.05])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Value")

    # Annotations
    def annotate_heatmap(ax, data, fmt="{:.2f}"):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", color="black")

    annotate_heatmap(ax1, np.real(rho))
    annotate_heatmap(ax2, np.real(rhoA))
    annotate_heatmap(ax3, np.real(rhoB))

    plt.savefig(savefig)
    plt.show()

def von_neumann_entropy(rho):
    """
    Compute the Von-Neumann entropy

    Args:
        rho (np.ndarray):
            Density matrix.

    Returns:
        float: 
            Von-Neumann entropy S(rho).
    """
    # Compute the eigenvalues
    eigvals = np.linalg.eigvalsh(rho)

    # Filter out zero eigenvalues to avoid log(0)
    eigvals = eigvals[eigvals > 1e-15] 

    # Return the Von-Neumann entropy
    return -np.sum(eigvals * np.log2(eigvals))


def benchmark(maxDim=10):
    """
    Benchmark time and memory usage for generating separable and general
    pure quantum states as a function of subsystem dimension.

    Args:
        maxDim (int):
            Maximum subsystem dimension to benchmark.

    Returns:
        tuple:
            time_separable_state (list[float]),
            memory_separable_state (list[float]),
            time_general_state (list[float]),
            memory_general_state (list[float])
    """

    # Store times
    time_separable_state = []
    time_general_state = []

    # Store memories
    memory_separable_state = []
    memory_general_state = []

    message_checkpoints(1, f"Benchmarking state generation up to dimension {maxDim}...", True)
    for d in range(2, maxDim+1):

        # SEPARABLE STATE
        # Time
        start_sep = time()
        nloops = 100
        for _ in np.arange(nloops): # loop to compute the mean
            _ = separable_pure_state([d, d])
        end_sep = time()
        time_sep = (end_sep - start_sep)/nloops
        time_separable_state.append(time_sep)

        # Memory
        psi_sep = separable_pure_state([d, d])
        size_bytes_sep = psi_sep.nbytes
        memory_separable_state.append(size_bytes_sep/1024)

        # GENERAL STATE
        # Time
        start_gen = time()
        for _ in np.arange(nloops): # loop to compute the mean
            _ = general_pure_state([d, d])
        end_gen = time()
        time_gen = (end_gen - start_gen)/nloops
        time_general_state.append(time_gen)

        # Memory
        psi_gen = general_pure_state([d, d])
        size_bytes_gen = psi_gen.nbytes
        memory_general_state.append(size_bytes_gen/1024)
        message_checkpoints(0, "Benchmarking done!", True)

    return time_separable_state, memory_separable_state, time_general_state, memory_general_state