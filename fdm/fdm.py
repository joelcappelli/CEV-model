import math
import numpy as np
from scipy.stats import norm

from qf.models.mkt_instrument_base import MktInstrument

"""
Application of a generic FDM applied to a parabolic PDE (Cauchy problem)
- theta [0,1] is a parameter allowing you to move from explicit to implicit difference scheme
- theta = 0 is implicit; theta = 1 is explicit
- note; theta = 0.5 is the Crank-Nicolson-scheme
See these notes for treatment of non-constant coefficients in a parabolic PDE
http://web.math.ku.dk/~rolf/teaching/ctff03/project2.pdf

See this paper for numerical method implementation specifically for CEV
http://pdf.xuebalib.com:1262/xuebalib.com.37179.pdf
"""

class FDM_Generic_CEV:
    def __init__(self,
                beta,
                mkt_instrument: MktInstrument,
                r,
                N,
                Nj,
                theta
                ):
        self._mkt_instrument = mkt_instrument
        self._spot = self._mkt_instrument.Spot
        self._sig = self._mkt_instrument.Q_vol
        self._cev_beta = beta
        self._K = self._mkt_instrument.Strike
        self._T = self._mkt_instrument.Maturity
        self._r = r
        self._N = N
        self._Nj = Nj
        self._theta = theta
        self._dt = self._T/self._N
        self._min_underlying = 0
        self._max_underlying = 2*self._K
        self._max_BC = self._mkt_instrument.PayOff(self._max_underlying) if self._mkt_instrument.PayOff(self._max_underlying) > 0  else self._mkt_instrument.PayOff(self._min_underlying)
        self._min_BC = 0
        self._dx = (self._max_underlying - self._spot)/self._Nj
        self._sol = np.zeros(2*self._Nj + 1)
        self._rhs = np.zeros(2*self._Nj + 1)
        self._initialise_tN_slide()
        self._update_tridiag(self._T)

    def _applyBC(self):
        self._gridslice[0] = 2.0*self._gridslice[1] - self._gridslice[2]
        self._gridslice[-1] = 2.0*self._gridslice[-2] - self._gridslice[-3]
        self._gridslice[self._gridslice < self._min_BC ] =  self._min_BC
        self._gridslice[self._gridslice > self._max_BC ] =  self._max_BC
        return

    def _Xj_applyConstraints(self, j: float):
        x = self._spot - self._Nj*self._dx + self._dx*j
        if x < self._min_underlying:
            x = self._min_underlying
        elif x > self._max_underlying:
            x = self._max_underlying
        return x

    def _initialise_tN_slide(self):
        self._gridslice = np.zeros(2*self._Nj + 1)
        tN_underlying = np.array([self._Xj_applyConstraints(j) for j in  range(0,2*self._Nj + 1)])
        for j in range(0, 2 * self._Nj + 1):
            self._gridslice[j] = self._mkt_instrument.PayOff(tN_underlying[j])
        self._applyBC()
        return

    def _mu_func(self, x,t):
        return self._r

    def _sig_func(self, x,t):
        return self._sig*(x**(self._cev_beta/2.0))

    def _a(self, x,t):
        mu = self._mu_func(x,t)
        sig = self._sig_func(x,t)
        return (1.0-self._theta)*(mu - sig*sig/self._dx)/(2.0*self._dx)

    def _b(self, x,t):
        sig = self._sig_func(x,t)
        return 1.0/self._dt + (1.0-self._theta)*(self._r +  sig*sig/(self._dx**2))

    def _c(self, x,t):
        mu = self._mu_func(x,t)
        sig = self._sig_func(x,t)
        return (1.0 - self._theta)*(-mu - sig*sig/self._dx)/(2.0*self._dx)

    def _alpha(self, x,t):
        mu = self._mu_func(x,t)
        sig = self._sig_func(x,t)
        return -self._theta*(mu - sig*sig/self._dx)/(2.0*self._dx)

    def _beta(self, x,t):
        sig = self._sig_func(x,t)
        return 1.0/self._dt - self._theta*(self._r +  sig*sig/(self._dx**2))

    def _gamma(self, x,t):
        mu = self._mu_func(x,t)
        sig = self._sig_func(x,t)
        return self._theta*(mu + sig*sig/self._dx)/(2.0*self._dx)

    def _update_tridiag(self, t):
        self._tridiag = np.zeros(shape = (2*self._Nj + 1,  2*self._Nj + 1))
        self._tridiag[0,0] = self._b(self._Xj_applyConstraints(0),t)
        self._tridiag[0,1] = self._c(self._Xj_applyConstraints(1),t)
        self._tridiag[-1,-2] = self._a(self._Xj_applyConstraints(2*self._Nj),t)
        self._tridiag[-1,-1] = self._b(self._Xj_applyConstraints(2*self._Nj + 1),t)
        for j in range(1,self._tridiag.shape[0]-1):
            x = self._Xj_applyConstraints(j)
            self._tridiag[j, j-1] = self._a(x,t)
            self._tridiag[j, j] = self._b(x,t)
            self._tridiag[j, j+1] = self._c(x,t)
        return

    def result(self):
        return self._gridslice[self._Nj]

    def rollback(self):
        t_from = self._N -1
        t_to = 0
        np.copyto(self._sol,self._gridslice)
        n = self._gridslice.shape[0]
        #work backwards to time starting from maturity/exercise date
        for i in range(t_from,t_to,-1):
            t = self._dt*i
            #for some time t, update the RHS to determine the  t - dt space grid
            for j in range(0, n-1):
                x = self._Xj_applyConstraints(j)
                self._rhs[j] = self._alpha(x, t)*self._sol[j-1] \
                                + self._beta(x, t)* self._sol[j] \
                                + self._gamma(x, t)* self._sol[j+1]

            self._sol =  np.linalg.solve(self._tridiag,self._rhs)
            np.copyto(self._gridslice,self._sol)
            self._applyBC()
            self._update_tridiag(t)
        return

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import qf.pricing_util.option as opt
    import qf.pricing_util.payoff as pf

    from qf.models.cev import CEV_Opt
    from qf.models.blackscholes import BS

    beta = 1.9999
    K = 40.0
    T = 0.5
    r = 0.05
    q = 0.0
    sig = 0.2
    theta = 0.5
    N = 300
    Nj = 300

    underlying_values = list(range(20,61,5))
    beta_values = [1.9999]#np.linspace(0.1,1.999,3).tolist()

    option = opt.EuropeanOption(pf.PayOffPut(strike=K), expiry=T)

    prices_fdm = {}
    prices_analy = {}

    for beta in beta_values:
        prices_fdm[beta] = []
        prices_analy[beta] = []

    prices_bs_equ = []

    payoff = []
    for S in underlying_values:
        print(f"Running for Underlying value: {S}")

        for beta in beta_values:

            print(f"Running for Beta: {beta}")

            # build the CEV instrument
            CEV_Euro = CEV_Opt(spot=S,
                              sig=sig,
                              beta=beta,
                              r=r,
                              q=q,
                              option=option
                              )

            FDM_Generic_engine = FDM_Generic_CEV(
                            beta=(CEV_Euro.Power*2.0),
                            mkt_instrument=CEV_Euro,
                            r=r,
                            N=N,
                            Nj=Nj,
                            theta=theta
                            )

            FDM_Generic_engine.rollback()
            prices_fdm[beta].append(FDM_Generic_engine.result())
            prices_analy[beta].append(CEV_Euro.Analytical_NPV())

        BS_Euro = BS(spot=S,
                     sig=sig,
                     r=r,
                     option=option
                     )

        prices_bs_equ.append(BS_Euro.Analytical_NPV())

        payoff.append(CEV_Euro.PayOff(S))

    fig, ax = plt.subplots()

    title_ = r'Comparison of FDM, Analytical and BS Option Price - CEV Process with varying $\beta$'

    line_handles = []

    for beta in beta_values:
        fdm_label = r'FDM $\theta$ =' + str(theta) + r' & $\beta$ = ' + str(beta)
        anal_label = r'Analy. $\beta$ = ' + str(beta)

        line_fdm, = ax.plot(np.array(underlying_values), np.array(prices_fdm[beta]), 'x-', label=fdm_label)
        line_analy, = ax.plot(np.array(underlying_values), np.array(prices_analy[beta]), 'o-', label=anal_label)
        line_handles.append(line_fdm)
        line_handles.append(line_analy)

    line_bs, = ax.plot(np.array(underlying_values), np.array(prices_bs_equ), 'b-', label = 'BS Equivalent')
    line_payoff, = ax.plot(np.array(underlying_values), np.array(payoff), 'k', label = 'Terminal Payoff')

    line_handles = line_handles + [line_bs, line_payoff]
    ax.legend(handles=line_handles)
    plt.grid()
    plt.xlabel('Underlying')
    plt.ylabel('Option Price')
    plt.title(title_)

    plt.show()
