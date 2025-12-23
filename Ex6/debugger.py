import numpy as np
from scipy.linalg import ishermitian, eigh


def message_checkpoints(verbosity, message, var=None, debug=True):
    """
    Checkpoint function to print messages.

    Args:
        verbosity (int): verbosity level of the function.
        message (str): message to print.
        var (int, float): variable to be printed.
        debug (bool, optional): decide to call the function or not. Defaults to True.
    """
    
    if debug == False:
        return 
    
    if verbosity==0:
        print(message)
    
    elif verbosity==1:
        print(message, var)
    
    else:
        raise TypeError(f"Verbosity level {verbosity} not found. Choose another verbosity level!")


def matrix_checkpoints(verbosity, A, debug=True):
    """
    Checkpoint function for matrix handling

    Args:
        verbosity (int): verbosity level of the function.
        A (float, float) or (complex, complex): matrix to test.
        debug (bool, optional): decide to call the function or not. Defaults to True.

    Raises:
        TypeError: no requirements fulfilled.

    Returns:
        bool: return True if the conditions are satisfyed.
    """

    if debug == False:
        return 
    
    # Define a tolerance for machine precision
    TOLERANCE = 1e-12
    
    if verbosity==0:
    # Check if a matrix is hermitian
        if (ishermitian(A)==True):
            print(f"You have correctly generated an hermitian matrix!")
            return True
        else:
            raise TypeError(f"ERROR: The matrix is not hermitian!")
        
    if verbosity==1:
    # Check if a matrix is diagonal
        if (np.count_nonzero(A - np.diag(np.diagonal(A))) == 0):
            print(f"You have correctly generated a diagonal matrix!")
            return True
        else: 
            raise TypeError(f"ERROR: The matrix is not diagonal!")
    
    if verbosity==2:
    # Check if a matrix is a density matrix
        
        # Check for the three conditions to be a density matrix:
        # - hermitian
        # - positive semidefinite
        # - Tr=1 

        if not (np.allclose(A, A.conj().T, atol=TOLERANCE)):
            raise TypeError(f"ERROR: The matrix is not a density matrix (it is not hermitian)!")
        
        # Compute eigenvalues
        eigvals, _ = eigh(A)
        
        if not (np.all(eigvals>=-TOLERANCE)):
            raise TypeError(f"ERROR: The matrix is not a density matrix (it is not positive semidefinite)!")
        
        if not (np.isclose(np.trace(A), 1.0, rtol=TOLERANCE)):
            raise TypeError(f"ERROR: The matrix is not a density matrix (trace is not 1)!")
        
        return True
    
    else:
        raise TypeError(f"Verbosity level {verbosity} not found. Choose another verbosity level!")
        

def vector_checkpoints(verbosity, v, w=None, dx=0.01, debug=True):
    """
    Checkpoint function for vector normalization check using discrete L2 norm.

    Args:
        verbosity (int): 0 for silent error raising, 1+ for print output.
        v (np.ndarray): The vector (eigenvector/eigenfunction) to check.
        dx (float): The spatial discretization step (self.dx) used for L2 norm.
        debug (bool, optional): Decides whether to run the check. Defaults to True.

    Raises:
        ValueError: If the vector is not normalized within tolerance.

    Returns:
        bool: True if the norm is close to 1.0.
    """

    if debug == False:
        return 
    
    # Define a tolerance for machine precision
    TOLERANCE = 1e-12
    
    if verbosity==0:
    # Check if a vector is normalized or not
        norm = np.linalg.norm(v)
        if np.isclose(norm, 1.0, atol=TOLERANCE):
            return True

        else:
            raise ValueError(f"ERROR: The vector is NOT normalized! Norm = {norm:.8f}. Expected 1.0.")


    elif verbosity==1:
    # Check if a vector is normalized or not in the discrete case
        norm = np.sqrt(np.sum(np.abs(v)**2) * dx)
        if np.isclose(norm, 1.0, atol=TOLERANCE):
            return True
        
        else:
            raise ValueError(f"ERROR: The vector is NOT normalized! Norm = {norm:.8f}. Expected 1.0.")
        
    
    elif verbosity==2:
    # Check orthonormality
        if w is None:
            raise ValueError("ERROR: you have to specify the value of w!")
        
        v = np.asarray(v)
        w = np.asarray(w)

        # Norms
        nv = np.linalg.norm(v)
        nw = np.linalg.norm(w)

        if not np.isclose(nv, 1.0, atol=TOLERANCE):
            raise ValueError("ERROR: norm of v is not 1")

        if not np.isclose(nw, 1.0, atol=TOLERANCE):
            raise ValueError("ERROR: norm of w is not 1")

        # Inner product 
        overlap = np.vdot(v, w)

        if not np.isclose(overlap, 0.0, atol=TOLERANCE):
            return False

        return True


    elif verbosity==3:
    # Check orthonormality in the discrete case
        if w is None:
            raise ValueError("ERROR: you have to specify the value of w!")
        
        v = np.asarray(v)
        w = np.asarray(w)

        # Norms
        nv = np.sqrt(np.sum(np.abs(v)**2)*dx)
        nw = np.sqrt(np.sum(np.abs(w)**2)*dx)

        if not np.isclose(nv, 1.0, atol=TOLERANCE):
            raise ValueError("ERROR: norm of v is not 1")

        if not np.isclose(nw, 1.0, atol=TOLERANCE):
            raise ValueError("ERROR: norm of w is not 1")

        # Inner product
        overlap = np.sum(np.conj(v) * w) * dx

        if not np.isclose(overlap, 0.0, atol=TOLERANCE):
            return False

        return True


    else:
        raise TypeError(f"Verbosity level {verbosity} not found. Choose another verbosity level!")