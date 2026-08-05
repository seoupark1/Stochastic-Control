import numpy as np

from stochastic_control.dynamics.rigid_body import RigidBody

def test_mrp_derivatives_are_zero_at_equilibrium():

    rigidbody = RigidBody(np.diag([100, 75, 80]))

    rotational_state = np.zeros(6)
    total_torque = np.zeros(3)

    derivatives = rigidbody.mrp_derivatives(rotational_state, total_torque)

    np.testing.assert_allclose(derivatives, np.zeros(6))