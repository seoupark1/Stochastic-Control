import numpy as np

from stochastic_control.math_tools import is_PSD
from stochastic_control.controllers.lqr.lti_infinite_horizon import InfiniteHorizonLQRController

def test_riccati_equation():

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)

    controller = InfiniteHorizonLQRController(A, B, Q, R)
    S = controller.S

    # test riccati equation's result is 0
    np.testing.assert_allclose(S @ A + A.T @ S - S @ B @ np.linalg.inv(R) @ B.T @ S + Q, np.zeros_like(Q), atol = 1e-10)

def test_is_S_PSD():

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)

    controller = InfiniteHorizonLQRController(A, B, Q, R)
    S = controller.S

    # test S is symmetric positive semi-definite
    assert is_PSD(S) is True