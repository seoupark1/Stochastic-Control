import numpy as np
from ..states.context import StateContext

class CombineDisturbances:

    def __init__(self,
                 models):

        self.models = models

    def torque(self,
               t: float,
               context: StateContext):

        total_torque = np.zeros(3)

        for models in self.models:
            model_torque = np.asarray(models.torque(t, context), dtype = float).reshape(3)
            total_torque += model_torque

        return total_torque