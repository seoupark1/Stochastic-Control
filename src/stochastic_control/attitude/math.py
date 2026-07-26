import numpy as np
from numpy.typing import ArrayLike, NDArray

# tilde operator
def skew_symmetric(v: ArrayLike) -> NDArray[np.float64]:

    v = np.asarray(v, dtype = float).reshape(3)
    result = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])

    return result