import numpy as np
import math
from scipy.stats import norm

from instrument_base import MktInstrument
from option import Option

class BS(MktInstrument):
    def __init__(self,spot: float,
                 sig: float,
                 r: float,
                 option: Option):
        self._spot = spot
        self._sig = sig
        self._r = r
        self._option = option
        self._cashflow_times = np.array([self._option.Exercise])
        self._initialise()

    @property
    def Spot(self):
        return self._spot

    @property
    def Q_drift(self):
        return self._r

    @property
    def Q_vol(self):
        return self._sig

    @property
    def CashflowTimes(self):
        return self._cashflow_times

    def _initialise(self):
        self._volsqrtT = self._sig*math.sqrt(self._option.Expiry)
        self._part_d1 = (self._r + (self._sig**2)/2.0)*self._option.Expiry
        self._discountedStrike = self._option.Strike*math.exp(-self._r*self._option.Expiry)

    def NPV(self, cashflow_times: np.ndarray, underlying_values: np.ndarray):
        #not ideal, equal comparison with float values
        terminal_value = underlying_values[np.where(cashflow_times == self._option.Exercise)]
        return self._option.PayOff(terminal_value)*math.exp(-self._r*self._option.Exercise)

    def Analytical_NPV(self):
        d1 = (math.log(self.Spot/self._option.Strike) + self._part_d1)/self._volsqrtT
        d2 = d1 - self._volsqrtT

        if self._option.PayOffType == 'call':
            return norm.cdf(d1)*self.Spot - norm.cdf(d2)*self._discountedStrike
        elif self._option.PayOffType == 'put':
            return -norm.cdf(-d1)*self.Spot + norm.cdf(-d2)*self._discountedStrike