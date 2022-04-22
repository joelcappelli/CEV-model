import numpy as np
import math
from scipy.stats import norm

from qf.models.mkt_instrument_base import MktInstrument
from qf.pricing_util.option import Option

class BS(MktInstrument):
    def __init__(self,spot: float,
                 sig: float,
                 r: float,
                 option: Option,
                 q = 0.0):
        MktInstrument.__init__(self,spot = spot)
        self._sig = sig
        self._r = r
        self._q = q
        self._option = option
        self._cashflow_times = np.array([self._option.Exercise])
        self._initialise()

    @property
    def Spot(self):
        return self._spot

    @property
    def Strike(self):
        return self._option.Strike

    @property
    def Maturity(self):
        return self._option.Expiry

    @property
    def Q_drift(self):
        return self._r - self._q

    @property
    def Q_vol(self):
        return self._sig

    @property
    def LogProcess_Q_drift(self):
        return self._r - self._q - 0.5*self._sig*self._sig

    @property
    def CashflowTimes(self):
        return self._cashflow_times

    def _initialise(self):
        self._volsqrtT = self._sig*math.sqrt(self._option.Exercise)
        self._part_d1 = (self._r- self._q + (self._sig**2)/2.0)*self._option.Exercise
        self._discountedStrike = self._option.Strike*math.exp(-(self._r - self._q)*self._option.Exercise)

    def PayOff(self, underlying: float):
        return self._option.PayOff(underlying)

    def NPV(self, realisation_times: np.ndarray, underlying_values: np.ndarray):
        #not ideal, equal comparison with float values
        terminal_value = underlying_values[np.where(realisation_times == self._option.Exercise)]
        return self._option.PayOff(terminal_value)*math.exp(-(self._r - self._q)*self._option.Exercise)

    def Analytical_NPV(self):
        d1 = (math.log(self.Spot/self._option.Strike) + self._part_d1)/self._volsqrtT
        d2 = d1 - self._volsqrtT

        if self._option.PayOffType == 'call':
            return norm.cdf(d1)*self.Spot - norm.cdf(d2)*self._discountedStrike
        elif self._option.PayOffType == 'put':
            return -norm.cdf(-d1)*self.Spot + norm.cdf(-d2)*self._discountedStrike

if __name__ == "__main__":
    from qf.pricing_util.option import EuropeanOption
    from qf.pricing_util.payoff import PayOffCall

    test_instrument = BS(spot=100,
                         sig=0.5,
                         r=0.05,
                         option=EuropeanOption(PayOffCall(strike=110), expiry=0.5)
                         )

    print('Analytical_NPV')
    print(test_instrument.Analytical_NPV())