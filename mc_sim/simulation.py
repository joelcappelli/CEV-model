import math
import numpy as np
from scipy.stats import norm

from sde.process_base import SDEProcess
from qf.models.mkt_instrument_base import MktInstrument

class SimulationConfig:

    @property
    def SnapshotSims(self):
        return self._snapshotsims

    @property
    def ConfidenceLevel(self):
        return self._CI

    @property
    def NumberSimus(self):
        return self._numberSimus

    @property
    def Goal(self):
        return self._goal

    def __init__(self, numberSimus, CI = 0.95, snapshotsims = 1000, goal = 0.05):
        self._numberSimus = numberSimus
        self._CI = CI
        self._snapshotsims = snapshotsims
        self._goal = goal


"""To do:
   - Fix the debug printing statements. Constant tab-width table view.     
"""
class SimStats:
    def __init__(self, numbersimus: int, CI: float, snapshotsims: int, goal: float, debug = False):
        self._results = np.zeros(numbersimus)
        self._simsDone = 0
        self._CI = CI
        self._CI_width = 0
        self._snapshotsims = snapshotsims
        self._snapshotStats = np.array([])
        self._goal= goal
        self._debug = debug

        assert (self._CI > 0.0) & (self._CI < 1.0),  f"CI must be > 0 and < 1, input was {self._CI}"

        if self._debug:
            print(f"Running simulation")
            print("{:<20} {:<20} {:<10}".format('Simulation Number  |', 'Simulation Output  |','CI width |'))

    def Store(self, sim: int, res: float):
        self._results[sim] = res
        self._simsDone = sim+1

        if self._debug:
            if self._simsDone%self._snapshotsims == 0:
                    print(f"{int(self.SimSnapshot[0]):15d} {self.SimSnapshot[1]:20.2f} {self.SimSnapshot[2]:12.4f}")
        return

    @property
    def AccuracyReached(self):
        if self.CI_width > 0:
            if self.CI_width < self._goal:
                return True
            else:
                return False
        else:
            return False

    @property
    def SimSnapshot(self):
        #store the number of simulations, most recent est, standard error range
        return np.array([self._simsDone, self.SimMean, self.CI_width])

    #To do: Add an Assert on 0 sims
    @property
    def SimMean(self):
        return np.sum(self._results)/float(self._simsDone)

    #To do: Add an Assert on 0 sims
    @property
    def SimVariance(self):
        return np.dot(self._results,self._results)/float(self._simsDone) - self.SimMean**2

    @property
    def CI_width(self):
        return 2*norm.ppf(self._CI)*math.sqrt(self.SimVariance/float(self._simsDone))

""" To do:
    - add a wrapper class for a portfolio of instruments 
    - add in numeraire process for real world 
    - extend for multiple underlyings
"""
class SimMapping:
    def __init__(self, underlying_process: SDEProcess, mkt_instrument: MktInstrument):
        self._mkt_instrument = mkt_instrument
        self._underlying_process = underlying_process

    def evaluate(self):
        cashflow_times,underlying_values = self._underlying_process.Xt(self._mkt_instrument.Spot ,self._mkt_instrument.CashflowTimes)
        return self._mkt_instrument.NPV(cashflow_times,underlying_values)

#Monte Carlo Simulation class
"""
Inputs: Simulation Config with a user configurable options
        - number of simulations, confidence level, standard error based goal, print out of on-going results
        Simulation mapping 
        - maps the simulated random process (risk factor/underlying values) to instruments so that discounted payoffs
        can be evaluated with every simulation
"""
class Simulation:

    @property
    def NbSimus(self):
        return self._nbSimus

    @property
    def SimuConfig(self):
        return self._simuConfig

    def __init__(self, simconfig: SimulationConfig, simmapping: SimMapping, debug = False):
        self._simconfig = simconfig
        self._simstats = SimStats(self._simconfig.NumberSimus, self._simconfig.ConfidenceLevel, self._simconfig.SnapshotSims, self._simconfig.Goal, debug)
        self._simMapping = simmapping
        self._nbSimus = self._simconfig.NumberSimus

    def run(self):
        for simidx in range(0, self._nbSimus):
            simOutput = self._simMapping.evaluate()
            self._simstats.Store(simidx, simOutput)
            if self._simstats.AccuracyReached:
                break

        return self._simstats.AccuracyReached, self._simstats.SimSnapshot

