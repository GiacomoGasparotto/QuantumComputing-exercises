import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from time import time

from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from debugger import *

def ising_hamiltonian(N, g=0, h=0):
    """
    Construct the Hamiltonian matrix for the 1D Transverse Field Ising Model.

    Args:
        N (int):
            The number of spins in the chain.
        g (float):
            The strength of the transverse magnetic field (coefficient of sigma_x).
            Default is 0.
        h (float):
            The strength of the longitudinal magnetic field (coefficient of sigma_z).
            Default is 0.

    Returns:
        scipy.sparse.csr_matrix:
            The Hamiltonian matrix of size (2^N, 2^N) in Compressed Sparse Row format.
    """

    # Define matrices
    Id = sparse.eye(2, dtype=complex) # Identity matrix
    sigmax = sparse.csr_matrix([[0, 1], 
                                [1, 0]], dtype=complex) # Pauli-X matrix
    sigmaz = sparse.csr_matrix([[1, 0], 
                                [0, -1]], dtype=complex) # Pauli-Z matrix
    
    # Hilbert space dimension
    d = 2**N

    # Initialize Hamiltonian terms
    transverse_field_term = sparse.csr_matrix((d, d), dtype=complex)
    interaction_term = sparse.csr_matrix((d, d), dtype=complex)
    longitudinal_field_term = sparse.csr_matrix((d, d), dtype=complex)    

    # Construct the Hamiltonian terms
    for i in range(N):
        # Transverse field term
        transverse_ops = [Id]*N
        transverse_ops[i] = sigmax
        transverse_field_term += reduce(sparse.kron, transverse_ops)

        # Interaction term
        if i!=(N-1):
            interaction_ops = [Id]*N
            interaction_ops[i] = sigmaz
            interaction_ops[(i+1)] = sigmaz
            interaction_term += reduce(sparse.kron, interaction_ops)

        # Longitudinal field term
        if h!=0:
            longitudinal_ops = [Id]*N
            longitudinal_ops[i] = sigmaz
            longitudinal_field_term += reduce(sparse.kron, longitudinal_ops)
    
    # Build and return the Hamiltonian
    H = - g*transverse_field_term - interaction_term - h*longitudinal_field_term

    matrix_checkpoints(0, H.toarray(), debug=False) 

    return H

def benchmark(maxDim=14, dense_limit=11, g=0.5, h=0):
    """
    Benchmark the time and memory performance of constructing the Ising Hamiltonian
    using Sparse matrices versus Dense matrices.

    Args:
        maxDim (int):
            The maximum number of spins (N) to simulate. Default is 14.
        dense_limit (int):
            The maximum N for which dense matrix calculations are attempted. 
            Used to prevent MemoryErrors (RAM overflow) for large N. Default is 11.
        g (float):
            Transverse field strength. Default is 0.5.
        h (float):
            Longitudinal field strength. Default is 0.

    Returns:
        dict:
            A dictionary containing lists of benchmark results:
            - "dims": List of N values tested.
            - "hilbert_dims": List of Hilbert space dimensions (2^N).
            - "time_sparse": Execution time (seconds) for sparse generation.
            - "mem_sparse": Memory usage (MB) for sparse matrices.
            - "time_dense": Execution time (seconds) for dense generation.
            - "mem_dense": Memory usage (MB) for dense matrices.
    """

    # Check for dense limit
    if dense_limit==None: dense_limit=maxDim

    # Dictionary to store all the results
    results = {
        "dims": [],         
        "hilbert_dims": [],  
        "time_sparse": [], 
        "mem_sparse": [],
        "time_dense": [],  
        "mem_dense": []
    }

    message_checkpoints(0, f"Benchmarking Sparse vs Dense...", True)

    # Loop over dimensions
    for N in range(2, maxDim + 1):
        message_checkpoints(0, f"Benchmarking for N: {N}/{maxDim}", True)
        
        # Store dimensions
        results["dims"].append(N)
        results["hilbert_dims"].append(2**N)

        # === SPARSE ===
        # Time
        start = time()
        H_sparse = ising_hamiltonian(N, g, h) 
        stop = time()
        t_sparse = stop - start
        
        # Memory
        mem_sparse_bytes = H_sparse.data.nbytes + H_sparse.indices.nbytes + H_sparse.indptr.nbytes
        
        # Store sparse results
        results["time_sparse"].append(t_sparse)
        results["mem_sparse"].append(mem_sparse_bytes / (1024**2)) # MB

        # === DENSE ===
        if N <= dense_limit:
            try:
                # Time
                start = time()
                # Convert to dense
                H_dense = H_sparse.toarray() 
                stop = time()
                t_dense = t_sparse + (stop - start)
                
                # Memory
                mem_dense_bytes = H_dense.nbytes
                
                # Store dense results
                results["time_dense"].append(t_dense)
                results["mem_dense"].append(mem_dense_bytes / (1024**2)) # MB

            except MemoryError:
                print(f"  Skipping Dense for N={N} (Memory Error)")

                # Store dense results
                results["time_dense"].append(np.nan)
                results["mem_dense"].append(np.nan)
        else:
            # Fill with NaN to keep exact dimensions
            results["time_dense"].append(np.nan)
            results["mem_dense"].append(np.nan)

    message_checkpoints(0, "Benchmarking done!", True)

    return results


def diagonalize_hamiltonian(H, k=10):
    """
    Compute the eigenvalues and eigenvectors of the Hamiltonian.
    
    This function automatically switches between a dense solver (`eigh`) and 
    a sparse solver (`eigsh`) based on the requested number of eigenvalues `k` 
    relative to the matrix dimension.

    Args:
        H (scipy.sparse.csr_matrix or numpy.ndarray):
            The Hamiltonian matrix.
        k (int):
            The number of eigenvalues/vectors to compute. Default is 10.

    Returns:
        tuple:
            - eigvals (np.ndarray): An array of the `k` smallest eigenvalues (sorted).
            - eigvecs (np.ndarray): A matrix where columns are the corresponding eigenvectors.
    """

    # Check for the Hamiltonian dimension
    dim = H.shape[0]

    if k>=(dim-1):
        eigvals, eigvecs = eigh(H.toarray())
        return eigvals[:k], eigvecs[:, :k]
    
    else:
        eigvals, eigvecs = eigsh(H, k=k, which="SA")
        return eigvals, eigvecs
    
def magnetization(N):
    """
    Construct the average longitudinal magnetization operator M_z.

    M_z = (1/N) * sum(sigma_z^i)

    Args:
        N (int):
            The number of spins in the chain.

    Returns:
        scipy.sparse.csr_matrix:
            The magnetization operator matrix of size (2^N, 2^N).
    """

    # Define matrices
    Id = sparse.eye(2, dtype=complex) # Identity matrix
    sigmaz = sparse.csr_matrix([[1, 0], 
                                [0, -1]], dtype=complex) # Pauli-Z matrix
    
    # Hilbert space dimension
    d = 2**N

    # Magnetization operator
    M = sparse.csr_matrix((d, d), dtype=complex)

    for i in range(N):
        ops = [Id]*N
        ops[i] = sigmaz
        M += reduce(sparse.kron, ops)

    final_M = M/N
    matrix_checkpoints(0, final_M.toarray(), debug=False) 

    return final_M

def ground_state(H):
    """
    Compute the Ground State (GS) wavefunction of the given Hamiltonian.

    Args:
        H (scipy.sparse.csr_matrix):
            The Hamiltonian matrix.

    Returns:
        np.ndarray:
            A 1D array representing the normalized Ground State eigenvector.
    """
    
    _, eigvecs = diagonalize_hamiltonian(H, k=1)
    GS = eigvecs[:, 0] # Extract the ground state

    vector_checkpoints(0, GS, debug=True)
    return GS