import numpy as np

class SDEProcess:
    def __init__(self, init_drift: float, init_vol: float):
        self._drift = init_drift
        self._vol = init_vol

    def Drift(self, t: float):
        pass

    def Vol(self, t: float):
        pass

    def Xt(self, X0: float, times: np.ndarray):
        pass