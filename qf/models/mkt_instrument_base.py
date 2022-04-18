import numpy as np

class MktInstrument:
    def __init__(self,spot: float):
        self._spot = spot

    @property
    def Spot(self):
        return self._spot

    @property
    def Q_drift(self):
        pass

    @property
    def Q_vol(self):
        pass

    @property
    def CashflowTimes(self):
        pass

    def NPV(self, cashflow_times: np.ndarray, underlying_values: np.ndarray):
        pass