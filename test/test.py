import unittest

import mc_sim.simulation as mc
import qf.pricing_util.option as opt
import qf.pricing_util.payoff as pf

from qf.models.blackscholes import BS
from qf.models.cev import CEV_Opt

from sde.gbm_process import GBM

class PayOffMethods(unittest.TestCase):
    def test_payoff_put(self):
        put_test = pf.PayOffPut(strike=5.0)
        self.assertEqual(put_test(10),0.0)
        self.assertEqual(put_test(2),3.0)
        
    def test_payoff_call(self):
        call_test = pf.PayOffCall(strike=5.0)
        self.assertEqual(call_test(13.0),8.0)
        self.assertEqual(call_test(3),0.0)

    def test_euro_call(self):
        call_test = pf.PayOffCall(strike=5.0)
        euro_call_test = opt.EuropeanOption(call_test, 1)
        self.assertEqual(euro_call_test.PayOff(13.0), 8.0)
        self.assertEqual(euro_call_test.PayOff(3), 0.0)
        self.assertEqual(euro_call_test.Strike,5.0)
        self.assertEqual(euro_call_test.Expiry,1)
        self.assertEqual(euro_call_test.PayOffType,"call")

    def test_euro_put(self):
        put_test = pf.PayOffPut(strike=5.0)
        euro_put_test = opt.EuropeanOption(put_test, 1)
        self.assertEqual(euro_put_test.PayOff(8),0.0)
        self.assertEqual(euro_put_test.PayOff(2),3.0)
        self.assertEqual(euro_put_test.Strike,5.0)
        self.assertEqual(euro_put_test.Expiry,1)
        self.assertEqual(euro_put_test.PayOffType,"put")

class SimulationMethods(unittest.TestCase):

    def test_simulation_GBM(self):
        #configure the simulation parameters
        config = mc.SimulationConfig(numberSimus=10000,
                                  snapshotsims=10000,
                                  CI=0.95,
                                  goal=1.0
                                  )

        #build the Black Scholes instrument
        test_instrument = BS(spot=100,
                          sig=0.5,
                          r=0.05,
                          option=opt.EuropeanOption(pf.PayOffCall(strike=110), expiry=0.5)
                          )

        #map the BS instrument to the GBM process for the simulation
        mapping = mc.SimMapping(underlying_process=GBM(drift=test_instrument.Q_drift,
                                                      vol=test_instrument.Q_vol
                                                      ),
                            mkt_instrument=test_instrument
                           )

        #create the simulation object
        sim = mc.Simulation(simconfig=config,
                        simmapping=mapping,
                        debug=False
                        )

        #run simuation and gather results
        sim_status, sim_snapshot = sim.run()
        est_npv = sim_snapshot[1]
        std_error = sim_snapshot[2]/2.0
        upper_bound_est = est_npv + std_error
        lower_bound_est = est_npv - std_error

        self.assertTrue(lower_bound_est <= test_instrument.Analytical_NPV() <= upper_bound_est)

class CEV(unittest.TestCase):

    # CEV price for European Option should converge to BS for beta -> 2 from below
    def test_analytical_CEV(self):
        sig = 0.2
        #choose a beta close to 2
        beta = 1.9999
        r = 0.05
        q = 0
        S = 30.0
        K = 30.0
        T = 1

        option = opt.EuropeanOption(pf.PayOffCall(strike=K), expiry=T)

        #build the CEV instrument
        inst_CEV = CEV_Opt(spot=S,
                       sig=sig,
                       beta=beta,
                       r=r,
                       q=q,
                       option=option
                       )

        #build the Black Scholes instrument
        inst_BS = BS(spot=S,
                     sig=sig,
                     r=r,
                     q=q,
                     option=option
                     )

        #compare NPV
        diff = abs(inst_CEV.Analytical_NPV() - inst_BS.Analytical_NPV())
        max_allowed_diff = 10**-3

        self.assertTrue(diff <= max_allowed_diff)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)


