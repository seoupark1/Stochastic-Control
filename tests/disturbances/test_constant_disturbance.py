import numpy as np
import pytest

from stochastic_control.disturbances.constant_disturbance import ConstantDisturbance

@pytest.mark.parametrize("t", [1.0, 2.0, 3.0, 4.0, 5.0])
def test_returns_the_same_disturbance(t):

    disturbance = ConstantDisturbance(np.array([5, 12, 97]))
    torque = disturbance.torque(t, None)

    np.testing.assert_allclose(torque, np.array([5, 12, 97]))