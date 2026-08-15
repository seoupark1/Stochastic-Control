import numpy as np
from numpy.typing import ArrayLike

# tilde operator
def skew_symmetric(v: ArrayLike):

    # check parameter
    v = np.asarray(v, dtype = float).reshape(3)

    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

# check symmetric positive definite
def is_SPD(matrix: ArrayLike):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric 
    if not np.allclose(matrix, matrix.T):
        return False

    # check positive definite
    try:
        np.linalg.cholesky(matrix)
        return True

    except np.linalg.LinAlgError:
        return False

# check symmetric positive semi-definite
def is_PSD(matrix: ArrayLike):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.all(eigenvalues >= 0):
        return True

    else:
        return False