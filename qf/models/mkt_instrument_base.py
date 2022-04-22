import numpy as np

class MktInstrument:
    def __init__(self,spot: float):
        self._spot = spot

    @property
    def Spot(self):
        return self._spot

    @property
    def Strike(self):
        pass

    @property
    def Maturity(self):
        pass

    @property
    def Q_drift(self):
        pass

    @property
    def Q_vol(self):
        pass

    @property
    def LogProcess_Q_drift(self):
        pass

    @property
    def CashflowTimes(self):
        pass

    def PayOff(self):
        pass

    def NPV(self, realisation_times: np.ndarray, underlying_values: np.ndarray):
        pass