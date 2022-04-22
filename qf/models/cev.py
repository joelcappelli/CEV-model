import math
from scipy.stats import norm
import numpy as np

from qf.models.mkt_instrument_base import MktInstrument
from qf.pricing_util.option import Option

"""
Approximations to the non-central chi-square distribution have been developed.
One particularly good approximation is derived by Sankaran (1963) - Sankaran, M., (1963), “Approximations to the Non-Central Chi-Square Distribution,”
Biometrika, 50, 199-204.

Extending on: Andersen, L. & Andreasen, J. (2000), ‘Volatility Skews and Extensions of the Libor Market Model’,
Applied Mathematical Finance 7 , 1–32
Theorem 3 - following these notes here
Analytical solution to European Call using the CEV model showing CEV equivalence
https://www.dropbox.com/s/bqalu6cihynb8ui/CEVequivalence.pdf?dl=0
"""
def nc_chi_squ_cdf(z: float, k: float, v: float):
    h = 1.0 - (2.0 / 3.0) * (v + k) * (3.0 * v + k)
    h /= ((2.0 * v + k) ** 2.0)
    p = (2.0 * v + k) / ((v + k) ** 2.0)
    m = (h - 1.0) * (1.0 - 3.0 * h)

    numer = h * p * (1.0 - h + (2.0 - h) * m * p / 2.0)
    numer = numer - 1.0+ (z / (v + k)) ** h
    denom = h * math.sqrt(2.0 * p * (1.0 + m * p))

    return norm.cdf(numer / denom)

"""
Schroder’s Formulation
- Schroder, M. (1989), ‘Computing the Constant Elasticity of Variance Option Pricing Formula’,
Journal of Finance
44
(1), 211–219.

CEV model
validate against following papers
https://ir.nctu.edu.tw/bitstream/11536/8273/1/000259723800005.pdf
https://arxiv.org/pdf/1803.10376.pdf
https://www3.nd.edu/~alindsa1/Publications/CEV_Lindsay_Brecher.pdf 
https://www.researchgate.net/publication/228423345_Valuation_of_standard_options_under_the_constant_elasticity_of_variance_model/link/0c9605212dd5e5eba9000000/download

"""
class CEV_Opt(MktInstrument):
    """
    CEV model
    """

    def __init__(self, spot: float,
                 sig: float,
                 beta: float,
                 r: float,
                 q: float,
                 option: Option):
        MktInstrument.__init__(self, spot)
        self._spot = spot

        self._sig = sig
        self._beta = beta
        self._r = r
        self._q = q
        self._option = option
        self._cashflow_times = np.array([self._option.Exercise])
        self._initialise()

        assert (self._beta >= 0.0) & (self._beta < 2.0),  f"Beta must be > 0 and < 2, input was {self._beta}"

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
    def Power(self):
        return self._beta/2.0

    @property
    def CashflowTimes(self):
        return self._cashflow_times

    def _compute_k(self):
        numer = 2.0*(self._r - self._q)
        denom = self._sig*self._sig*(2.0 - self._beta)*(math.exp((self._r - self._q)*self._option.Exercise*(2.0 - self._beta))-1.0)
        return numer / denom

    def _compute_x(self):
        x = self._k*(self._spot ** (2.0 - self._beta))*math.exp((self._r - self._q)*self._option.Exercise*(2.0 - self._beta))
        return x

    def _compute_y(self):
        y = self._k*(self._option.Strike**(2.0 - self._beta))
        return y

    def _initialise(self):
        self._k = self._compute_k()
        self._x = self._compute_x()
        self._y = self._compute_y()

    def PayOff(self, underlying: float):
        return self._option.PayOff(underlying)

    def Analytical_NPV(self):
        two_on_two_minus_beta = 2.0/(2.0 - self._beta)
        if self._option.PayOffType == 'call':
            return self._spot * math.exp(-self._q*self._option.Exercise) * (1.0-nc_chi_squ_cdf(2.0*self._y, 2.0 + two_on_two_minus_beta,2.0*self._x)) \
                   - self._option.Strike * math.exp(-self._r * self._option.Exercise) * (nc_chi_squ_cdf(2.0*self._x, two_on_two_minus_beta, 2.0*self._y))
        elif self._option.PayOffType == 'put':
            return -self._spot * math.exp(-self._q*self._option.Exercise) * (nc_chi_squ_cdf(2.0*self._y, 2.0 + two_on_two_minus_beta,2.0*self._x)) \
                   + self._option.Strike * math.exp(-self._r * self._option.Exercise) * (1.0-nc_chi_squ_cdf(2.0*self._x, two_on_two_minus_beta, 2.0*self._y))

    def NPV(self, cashflow_times: np.ndarray, underlying_values: np.ndarray):
        # not ideal, equal comparison with float values
        terminal_value = underlying_values[np.where(cashflow_times == self._option.Exercise)]
        return self._option.PayOff(terminal_value) * math.exp(-self._r * self._option.Exercise)

if __name__ == "__main__":
    from qf.pricing_util.option import EuropeanOption
    from qf.pricing_util.payoff import PayOffCall, PayOffPut
    from qf.models.blackscholes import BS

    sig = 0.2
    beta = 1.9999
    r = 0.05
    q = 0
    S = 30.0
    K = 30.0
    T = 1

    opt = EuropeanOption(PayOffPut(strike=K), expiry=T)

    test_BS = BS(spot=S,
                 sig=sig,
                 r=r,
                 q=q,
                 option=opt
                 )

    print('Analytical_NPV - BS')
    print(test_BS.Analytical_NPV())

    test_CEV = CEV_Opt(spot=S,
                    sig=sig,
                    beta=beta,
                    r=r,
                    q=q,
                    option=opt
                    )

    print('Analytical_NPV - CEV')
    print(test_CEV.Analytical_NPV())