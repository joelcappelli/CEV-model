import math
import numpy as np

from .process_base import SDEProcess

class GBM(SDEProcess):
    def __init__(self, drift: float, vol: float):
        SDEProcess.__init__(self,init_drift = drift,init_vol = vol)

    @property
    def Drift(self):
        return self._drift

    @property
    def Vol(self):
        return self._vol

    """Terminal value
    """
    def Xt(self, X0: float, times: np.ndarray):
        def X_t(T):
            return X0 * math.exp((self.Drift - self.Vol*self.Vol/2.0)* T + self.Vol * math.sqrt(T) * np.random.normal())

        npX_t = np.vectorize(X_t)

        return times, npX_t(times)

class SimGBM(SDEProcess):
    def __init__(self, drift: float, vol: float):
        SDEProcess.__init__(self,init_drift = drift,init_vol = vol)

    @property
    def Drift(self):
        return self._drift

    @property
    def Vol(self):
        return self._vol

    """Terminal value
    """
    def Xt(self, X0: float, times: np.ndarray):
        #Not ideal, think about adding in a Scheme class to adjust the stepsize
        dt = 0.01
        dt_sqrt = math.sqrt(dt)

        """To do: Update to ensure the process is simulated for the specific cashflow times
            Currently assumes dt is small enough that if you simulate to max time, there will be times
            sufficiently close to all of the cashflow times
        """
        max_sim_time = np.max(times)
        nbTSteps = int(max_sim_time/float(dt))
        sim_times = np.linspace(start=dt,stop=max_sim_time,num=nbTSteps)

        # randomness generator
        dW_t = np.random.normal(size=(nbTSteps)) * dt_sqrt
        X_t = np.zeros(nbTSteps)
        X_t[0] = X0

        for timeidx in range(1, nbTSteps):
            X_t[timeidx] = X_t[timeidx - 1] + X_t[timeidx - 1]*self.Drift * dt + X_t[timeidx - 1]*self.Vol * dW_t[timeidx - 1]

        return sim_times,X_t