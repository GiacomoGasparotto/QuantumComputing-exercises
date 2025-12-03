import numpy as np
from scipy.linalg import ishermitian


def message_checkpoints(verbosity, message, debug=True):
    """
    Checkpoint function to print messages.

    Args:
        verbosity (int): verbosity level of the function.
        message (str): message to print.
        debug (bool, optional): decide to call the function or not. Defaults to True.
    """
    
    if debug == False:
        return 
    
    if verbosity==0:
        print(message)
    
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
    
    else:
        raise TypeError(f"Verbosity level {verbosity} not found. Choose another verbosity level!")
        
import numpy as np

def vector_checkpoints(verbosity, v, dx, debug=True):
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

    if not debug:
        return True
    
    TOLERANCE = 1e-8
    norm_discrete = np.sqrt(np.sum(np.abs(v)**2) * dx)
    is_normalized = np.isclose(norm_discrete, 1.0, atol=TOLERANCE)
    
    if is_normalized:
        if verbosity >= 1:
            print(f"PASS: Vector is correctly normalized! Norm = {norm_discrete:.8f}")
        return True
    else:
        raise ValueError(f"ERROR: The vector is NOT normalized! Norm = {norm_discrete:.8f}. Expected 1.0.")