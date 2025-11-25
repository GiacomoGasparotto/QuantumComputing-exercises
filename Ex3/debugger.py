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
