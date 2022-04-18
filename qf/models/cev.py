import math
from scipy.stats import norm
import numpy as np

from .mkt_instrument_base import MktInstrument
from qf.pricing_util.option import Option

"""
CEV model
#analytical solution to European Call using the CEV model
# https://www.dropbox.com/s/bqalu6cihynb8ui/CEVequivalence.pdf?dl=0

To do:
- double check the non-central chi squ. distribution approximation
 (see https://keisan.casio.com/exec/system/1180573183)
- check analytical npv formula with https://web.archive.org/web/20180220152049/http://rafaelmendoza.org/wp-content/uploads/2014/01/CEVchapter.pdf
http://www.delamotte-b.fr/CEV.aspx
https://arxiv.org/pdf/1803.10376.pdf
https://www3.nd.edu/~alindsa1/Publications/CEV_Lindsay_Brecher.pdf 
"""
def chi_squ_approx(z: float, k: float, v: float):
    h = 1 - (2.0 / 3.0) * (v + k) * (3 * v + k)
    h = h / ((2 * v + k) ** 2)
    p = (2 * v + k) / ((v + k) ** 2)
    m = (h - 1) * (1 - 3 * h)

    numer = h * p * (1 - h + (2 - h) * m * p / 2)
    numer = numer - 1 + (z / (v + k)) ** h
    denom = h * math.sqrt(2 * p * (1 + m * p))

    return norm.cdf(numer / denom)


class SimpleCEV(MktInstrument):
    """
    CEV model
    """

    def __init__(self, spot: float,
                 lmbda: float,
                 beta: float,
                 option: Option):
        self._spot = spot
        self._lmbda = lmbda
        self._beta = beta
        self._option = option
        self._cashflow_times = np.array([self._option.Exercise])
        self._vT = self._lmbda * self._lmbda * self._option.Expiry
        self._initialise()

        assert (self._beta >= 0) & (self._beta < 1), f"0 < beta < 1 expected, got: {self._beta}"

    @property
    def Spot(self):
        return self._spot

    @property
    def Q_drift(self):
        return 0

    @property
    def Q_vol(self):
        return self._lmbda

    @property
    def CashflowTimes(self):
        return self._cashflow_times

    def _compute_c(self):
        numer = self._spot ** (2 * (1 - self._beta))
        denom = (1 - self._beta) * (1 - self._beta) * self._vT
        return numer / denom

    def _compute_b(self):
        return 1 / (1 - self._beta)

    def _compute_a(self):
        numer = self._option.Strike ** (2 * (1 - self._beta))
        denom = (1 - self._beta) * (1 - self._beta) * self._vT
        return numer / denom

    def _initialise(self):
        self._param_a = self._compute_a()
        self._param_b = self._compute_b()
        self._param_c = self._compute_c()

    def Analytical_NPV(self):
        if self._option.PayOffType == 'call':
            return self._spot * (1 - chi_squ_approx(self._param_a, self._param_b + 2, self._param_c)) - self._option.Strike * chi_squ_approx(self._param_c, self._param_b, self._param_a)
        elif self._option.PayOffType == 'put':
            return -self._spot * chi_squ_approx(self._param_a,self._param_b + 2,self._param_c) + self._option.Strike * (1 - chi_squ_approx(self._param_c, self._param_b, self._param_a))

    def NPV(self, cashflow_times: np.ndarray, underlying_values: np.ndarray):
        #not ideal, equal comparison with float values
        terminal_value = underlying_values[np.where(cashflow_times == self._option.Exercise)]
        return self._option.PayOff(terminal_value)*math.exp(-self._r*self._option.Exercise)

class CEV:
    """
    CEV model
    """
    def __init__(self, spot: float,
                 lmbda: float, 
                 beta: float, 
                 r: float, 
                 q: float, 
                 option: Option):
        self._spot = spot
        self._lmbda = lmbda
        self._beta = beta
        self._r = r
        self._q = q
        self._option = option
        self._cashflow_times = np.array([self._option.Exercise])
        self._initialise() 

        assert (self._beta >= 0) & (self._beta < 1), f"0 < beta < 1 expected, got: {self._beta}"

    @property
    def Spot(self):
        return self._spot

    @property
    def Q_drift(self):
        return self._r - self._q

    @property
    def Q_vol(self):
        return self._lmbda

    @property
    def CashflowTimes(self):
        return self._cashflow_times

    def _compute_vT(self):
        if self._r != self._q:
            numer = self._lmbda*self._lmbda*(math.exp(2*(self._r - self._q)*(self._beta-1)*self._option.Expiry) - 1)
            denom = 2*(self._r - self._q)*(self._beta-1)
            vT = numer/denom
        else:
            vT = self._lmbda*self._lmbda*self._option.Expiry

        return vT

    def _compute_c(self):
        numer = self._spot**(2*(1-self._beta))
        denom = (1-self._beta)*(1-self._beta)*self._vT
        return numer/denom
        
    def _compute_b(self):
        return 1/abs(1-self._beta)
        
    def _compute_a(self):
        numer = (self._option.Strike*math.exp(-(self._r - self._q)*self._option.Expiry))**(2*(1-self._beta))
        denom = (1-self._beta)*(1-self._beta)*self._vT
        return numer/denom
    
    def _initialise(self):
        self._vT = self._compute_vT()
        self._param_a = self._compute_a()
        self._param_b = self._compute_b()
        self._param_c = self._compute_c()

    def Analytical_NPV(self):
        if self._option.PayOffType == 'call':
            return self._spot*math.exp(-self._q*self._option.Expiry)*(1- chi_squ_approx(self._param_a, self._param_a + self._param_b, self._param_c)) - self._option.Strike*math.exp(-self._r*self._option.Expiry)*chi_squ_approx(self._param_c, self._param_b, self._param_a)
        elif self._option.PayOffType == 'put':
            return -self._spot*math.exp(-self._q*self._option.Expiry)*chi_squ_approx(self._param_a, self._param_a + self._param_b, self._param_c) + self._option.Strike*math.exp(-self._r*self._option.Expiry)*(1-chi_squ_approx(self._param_c, self._param_b, self._param_a))

    def NPV(self, cashflow_times: np.ndarray, underlying_values: np.ndarray):
        #not ideal, equal comparison with float values
        terminal_value = underlying_values[np.where(cashflow_times == self._option.Exercise)]
        return self._option.PayOff(terminal_value)*math.exp(-self._r*self._option.Exercise)
        

