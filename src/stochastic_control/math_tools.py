import numpy as np
from numpy.typing import ArrayLike

# tilde operator
def skew_symmetric(v: ArrayLike):

    v = np.asarray(v, dtype = float).reshape(3)
    result = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

    return result

# check symmetric positive definite
def is_SPD(matrix):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric positive definite
    if matrix == matrix.T and (np.linalg.cholesky(matrix) is True):
        return True

    else:
        return False

# check symmetric positive semi-definite
def is_PSD(matrix):

    # check parameter
    matrix = np.asarray(matrix, dtype = float)

    # check symmetric positive semi-definite
    eigenvalues = np.linalg.eigvals(matrix)
    if matrix == matrix.T and (np.all(eigenvalues > 0) is True):
        return True

    else:
        return False