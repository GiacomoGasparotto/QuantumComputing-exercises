import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from functools import reduce

from debugger import *

def ising_hamiltonian(N, g=0, h=0):
    """
    Constructs the exact Hamiltonian matrix for the 1D Transverse Field Ising Model.

    The Hamiltonian is defined as: H = g*Sum(Sx) + Sum(SzSz) + h*Sum(Sz).

    Args:
        N (int): Number of lattice sites.
        g (float, optional): Transverse magnetic field strength. Defaults to 0.
        h (float, optional): Longitudinal magnetic field strength. Defaults to 0.

    Returns:
        scipy.sparse.csr_matrix: The Hamiltonian matrix of size (2^N, 2^N) in CSR format.
    """

    Id = sparse.eye(2, dtype=complex)
    sigmax = sparse.csr_matrix([[0, 1], 
                                [1, 0]], dtype=complex)
    sigmaz = sparse.csr_matrix([[1, 0], 
                                [0, -1]], dtype=complex)
    
    d = 2**N
    transverse_field_term = sparse.csr_matrix((d, d), dtype=complex)
    interaction_term = sparse.csr_matrix((d, d), dtype=complex)
    longitudinal_field_term = sparse.csr_matrix((d, d), dtype=complex)    

    for i in range(N):
        # Transverse field
        transverse_ops = [Id]*N
        transverse_ops[i] = sigmax
        transverse_field_term += reduce(sparse.kron, transverse_ops)

        # Interaction
        if i != (N-1):
            interaction_ops = [Id]*N
            interaction_ops[i] = sigmaz
            interaction_ops[(i+1)] = sigmaz
            interaction_term += reduce(sparse.kron, interaction_ops)

        # Longitudinal field
        if h != 0:
            longitudinal_ops = [Id]*N
            longitudinal_ops[i] = sigmaz
            longitudinal_field_term += reduce(sparse.kron, longitudinal_ops)
    
    H = g*transverse_field_term + interaction_term + h*longitudinal_field_term
    return H

# =========== RSRG algorithm ===========

def RSRG_init(twoN, g=0, h=0):
    """
    Initializes the Real-Space Renormalization Group (RSRG) algorithm configuration.

    It constructs the initial Left and Right blocks and the total superblock Hamiltonian
    for a starting system size.

    Args:
        twoN (int): The total size of the initial superblock. Must be even (will be incremented if odd).
        g (float, optional): Transverse magnetic field strength. Defaults to 0.
        h (float, optional): Longitudinal magnetic field strength. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - H_L (scipy.sparse.csr_matrix): Hamiltonian of the Left block.
            - H_R (scipy.sparse.csr_matrix): Hamiltonian of the Right block.
            - A (scipy.sparse.csr_matrix): Interaction operator on the right edge of the Left block.
            - B (scipy.sparse.csr_matrix): Interaction operator on the left edge of the Right block.
            - H0 (scipy.sparse.csr_matrix): The total superblock Hamiltonian.
    """

    if twoN % 2 != 0:
        twoN += 1
    
    N = twoN // 2
    Id2 = sparse.eye(2, dtype=complex)
    sigmaz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    
    # H_L and H_R (Size 2^N)
    H_sub = ising_hamiltonian(N, g, h)
    d = 2**N
    Id_sub = sparse.eye(d, dtype=complex)

    H_L = sparse.kron(H_sub, Id_sub)
    H_R = sparse.kron(Id_sub, H_sub)
    
    # Interaction Operators 
    A_ops = [Id2]*(N-1) + [sigmaz]
    A = reduce(sparse.kron, A_ops)

    B_ops = [sigmaz] + [Id2]*(N-1)
    B = reduce(sparse.kron, B_ops)

    interaction_term = sparse.kron(A, B)
    H0 = H_L + H_R + interaction_term

    matrix_checkpoints(1, H0.toarray(), debug=False)

    return H_L, H_R, A, B, H0

def RSRG_update(H_L, H_R, A, B, H0):
    """
    Performs a single Real-Space Renormalization Group (RSRG) decimation step.

    This function diagonalizes the superblock Hamiltonian, selects the lowest energy eigenstates,
    and projects the operators onto this truncated basis to form the next iteration's blocks.

    Args:
        H_L (scipy.sparse.csr_matrix): Current Left block Hamiltonian.
        H_R (scipy.sparse.csr_matrix): Current Right block Hamiltonian.
        A (scipy.sparse.csr_matrix): Current Left block interaction operator.
        B (scipy.sparse.csr_matrix): Current Right block interaction operator.
        H0 (scipy.sparse.csr_matrix): Current Superblock Hamiltonian.

    Returns:
        tuple: A tuple containing:
            - E0 (float): The ground state energy correction to be accumulated.
            - H_L_next (scipy.sparse.csr_matrix): Renormalized Hamiltonian for the next Left block.
            - H_R_next (scipy.sparse.csr_matrix): Renormalized Hamiltonian for the next Right block.
            - A_new (scipy.sparse.csr_matrix): Renormalized interaction operator for the next step.
            - B_new (scipy.sparse.csr_matrix): Renormalized interaction operator for the next step.
            - H_next (scipy.sparse.csr_matrix): The new Superblock Hamiltonian for the next step.
    """

    # Determine Dimensions
    dim_super = H0.shape[0]
    dim_sub = int(np.sqrt(dim_super))
    Id_sub = sparse.eye(dim_sub, dtype=complex)

    # Diagonalize
    eigvals, eigvecs = eigsh(H0, k=2, which="SA")    

    P = eigvecs 
    E0 = eigvals[0]
    E1 = eigvals[1]

    vector_checkpoints(0, P[:, 0], debug=True) # Normalization check

    # Renormalize Block Hamiltonian
    H_block_new = sparse.diags([0, E1 - E0], format='csr')

    matrix_checkpoints(1, H_block_new.toarray(), debug=False) 

    # Project the operators to the new basis.
    A_expanded = sparse.kron(Id_sub, A) 
    A_new = P.conj().T @ A_expanded @ P
    B_expanded = sparse.kron(B, Id_sub)
    B_new = P.conj().T @ B_expanded @ P

    # Construct Terms for the NEXT Superblock 
    Id_new = sparse.eye(2, dtype=complex)
    
    H_L_next = sparse.kron(H_block_new, Id_new)
    H_R_next = sparse.kron(Id_new, H_block_new)
    interaction_next = sparse.kron(A_new, B_new)

    H_next = H_L_next + H_R_next + interaction_next

    return E0, H_L_next, H_R_next, A_new, B_new, H_next

# ========== DMRG algorithm ==========

def iDMRG_init(g=0):
    """
    Initializes the Infinite Density Matrix Renormalization Group (iDMRG) algorithm.

    Sets up the initial single-site block Hamiltonian and boundary operator.

    Args:
        g (float, optional): Transverse magnetic field strength. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - Hm (scipy.sparse.csr_matrix): Initial single-site Hamiltonian.
            - A (scipy.sparse.csr_matrix): Initial boundary operator (Sigma-Z).
    """

    sigmax = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    sigmaz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)

    # Initialize Hamiltonian 
    Hm = g * sigmax 
    A = sigmaz

    return Hm, A

def iDMRG_update(Hm, A, g, m):
    """
    Performs a single iDMRG growth and truncation step.

    The function grows the system by adding a site, forms the superblock, computes the
    ground state, determines the reduced density matrix, and truncates the basis
    keeping the 'm' most significant states.

    Args:
        Hm (scipy.sparse.csr_matrix): Current block Hamiltonian.
        A (scipy.sparse.csr_matrix): Current boundary operator.
        g (float): Transverse magnetic field strength.
        m (int): Bond dimension (maximum number of states to keep).

    Returns:
        tuple: A tuple containing:
            - Hm_new (scipy.sparse.csr_matrix): The renormalized block Hamiltonian (grown by 1 site).
            - A_new_proj (scipy.sparse.csr_matrix): The renormalized boundary operator.
            - E_GS (float): The total ground state energy of the current superblock.
    """
    
    # Define single site operators 
    Id_site = sparse.eye(2, dtype=complex)
    sigmax = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    sigmaz = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    
    # Block Dimension
    d = Hm.shape[0]
    Id_block = sparse.eye(d, dtype=complex)

    # Grow Left Block 
    H_L = sparse.kron(Hm, Id_site) + sparse.kron(Id_block, g*sigmax) + sparse.kron(A, sigmaz)

    # New boundary operator 
    A_new = sparse.kron(Id_block, sigmaz)

    # Def. Superblock 
    d_new = H_L.shape[0]
    Id_new = sparse.eye(d_new, dtype=complex)
    H = sparse.kron(H_L, Id_new) + sparse.kron(Id_new, H_L) + sparse.kron(A_new, A_new)

    matrix_checkpoints(0, H.toarray(), debug=False)

    # Diagonalize
    eigvals, eigvecs = eigsh(H, k=1, which="SA")
    E_GS = eigvals[0]
    GS = eigvecs[:, 0]  

    vector_checkpoints(0, GS, debug=True) # normalization check

    # Create density matrix
    GS = GS.reshape((d_new, d_new))
    rho = GS @ GS.conj().T    

    matrix_checkpoints(2, rho, debug=True) # density matrix check

    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    
    # Sort descending
    idx = np.argsort(eigvals_rho)[::-1]
    n_keep = min(m, d_new)
    eigvecs_rho = eigvecs_rho[:, idx]
    P = eigvecs_rho[:, :n_keep] 

    # Apply projection
    Hm_new = P.conj().T @ H_L @ P   
    A_new_proj = P.conj().T @ A_new @ P

    return Hm_new, A_new_proj, E_GS