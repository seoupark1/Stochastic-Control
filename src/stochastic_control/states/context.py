from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen = True)
class StateContext:
    position_N: NDArray[np.float64] | None = None
    velocity_N: NDArray[np.float64] | None = None
    attitude_BN: NDArray[np.float64] | None = None
    angular_velocity_BN: NDArray[np.float64] | None = None