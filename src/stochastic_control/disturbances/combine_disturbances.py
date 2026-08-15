import numpy as np
from ..providers.body_state import BodyStateContext

class CombineDisturbances:

    def __init__(self,
                 models):

        self.models = models

    def torque(self,
               t: float,
               context: BodyStateContext):

        total_torque = np.zeros(3)

        for models in self.models:
            model_torque = np.asarray(models.torque(t, context), dtype = float).reshape(3)
            total_torque += model_torque

        return total_torque