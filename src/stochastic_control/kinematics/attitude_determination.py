import numpy as np
from numpy.typing import ArrayLike, NDArray
from attitude import crp_to_dcm

def skew_symmetric(v):
    v = np.array(v).flatten()
    result = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

    return result

# Triaxial Attitude Determination (TRIAD)
# Assume sensor1 is more accurate than sensor2
def triad_method(v1_B, v2_B, v1_N, v2_N):

    # normalization
    v1_B = v1_B / np.linalg.norm(v1_B)
    v2_B = v2_B / np.linalg.norm(v2_B)
    v1_N = v1_N / np.linalg.norm(v1_N)
    v2_N = v2_N / np.linalg.norm(v2_N)

    # define t
    t1_B = v1_B
    t2_B = skew_symmetric(v1_B) @ v2_B / np.linalg.norm(skew_symmetric(v1_B) @ v2_B)
    t3_B = skew_symmetric(t1_B) @ t2_B

    t1_N = v1_N
    t2_N = skew_symmetric(v1_N) @ v2_N / np.linalg.norm(skew_symmetric(v1_N) @ v2_N)
    t3_N = skew_symmetric(t1_N) @ t2_N

    # get dcm_estimate
    dcm_BbarT = np.column_stack((t1_B, t2_B, t3_B))
    dcm_NT = np.column_stack((t1_N, t2_N, t3_N))
    dcm_BbarN = dcm_BbarT @ np.transpose(dcm_NT)

    return dcm_BbarN

# Davenport's q-method about N sensors
def q_method(b_vectors: ArrayLike, n_vectors: ArrayLike, weight_vectors: ArrayLike) -> NDArray[np.float64]:

    # dimension validation
    assert len(b_vectors) == len(n_vectors) == len(weight_vectors), "Error: The number of data pairs does not match n"

    # data processing
    n_sensors = len(b_vectors)
    measured_vectors = np.asarray(b_vectors, dtype = float).reshape(n_sensors, 3)
    reference_vectors = np.asarray(n_vectors, dtype = float).reshape(n_sensors, 3)
    weights = np.asarray(weight_vectors, dtype = float).reshape(n_sensors, 1)
    b_matrix = np.zeros(3)

    # normalization
    measured_vectors /= np.linalg.norm(measured_vectors, axis=1, keepdims=True)
    reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)

    # get b_matrix
    b_matrix = (weights * measured_vectors).T @ reference_vectors

    # initial parameters
    z_matrix = np.array([[b_matrix[1,2] - b_matrix[2,1]], 
                         [b_matrix[2,0] - b_matrix[0,2]],
                         [b_matrix[0,1] - b_matrix[1,0]]])
    sigma = np.trace(b_matrix)
    s_matrix = b_matrix + b_matrix.T
    k_matrix = np.block([[sigma, z_matrix.T], [z_matrix, s_matrix - sigma * np.eye(3)]])

    # finding k_matrix's largest eigenvector which is optimal quaternion
    eigenvalues, eigenvectors = np.linalg.eigh(k_matrix)
    optimal_quaternion = eigenvectors[:, 3]

    return optimal_quaternion

# optimal linear attitude estimator (olae)
def olae_method(b_vectors: ArrayLike, n_vectors: ArrayLike, weight_vectors: ArrayLike) -> NDArray[np.float64]:

    # dimension validation
    assert len(b_vectors) == len(n_vectors) == len(weight_vectors), "Error: The number of data pairs does not match n"

    # data processing
    n_sensors = len(b_vectors)
    measured_vectors = np.asarray(b_vectors, dtype = float).reshape(n_sensors, 3)
    reference_vectors = np.asarray(n_vectors, dtype = float).reshape(n_sensors, 3)
    weights = np.asarray(weight_vectors, dtype = float).reshape(n_sensors, 1)

    # normalization
    measured_vectors /= np.linalg.norm(measured_vectors, axis=1, keepdims=True)
    reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)

    # initial parameters
    d_vectors = measured_vectors - reference_vectors
    d_matrix = d_vectors.reshape(3 * n_sensors, 1)

    s_vectors = measured_vectors + reference_vectors
    s_matrix = np.zeros((3 * n_sensors, 3))

    w_matrix = np.zeros(3 * n_sensors, 3 * n_sensors)

    for i in range(n_sensors):
        s_matrix[3*i:3*(i+1), :] = skew_symmetric(s_vectors[i, :])
        w_matrix[3*i:3*(i+1), 3*i:3*(i+1)] = weights[i, 0] * np.eye(3)

    # derive optimal dcm
    optimal_crp = np.linalg.inv(s_matrix.T @ w_matrix @ s_matrix) @ s_matrix.T @ w_matrix @ d_matrix
    optimal_dcm = crp_to_dcm(optimal_crp.flatten())

    return optimal_dcm








