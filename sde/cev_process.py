import math
import numpy as np

from .process_base import SDEProcess

class CEV(SDEProcess):
    def __init__(self, drift: float, vol: float, power: float):
        SDEProcess.__init__(self,init_drift = drift,init_vol = vol)
        self._power = power

    @property
    def Drift(self):
        return self._drift

    @property
    def Vol(self):
        return self._vol

    @property
    def Power(self):
        return self._power

    """Terminal value
    """
    def Xt(self, X0: float, times: np.ndarray):
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
            X_t[timeidx] = X_t[timeidx - 1] + X_t[timeidx - 1]*self.Drift * dt + (X_t[timeidx - 1]**self.Power)*self.Vol * dW_t[timeidx - 1]

        return sim_times,X_t