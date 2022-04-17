import unittest

from simulation import *
from blackscholes import *
from option import *
from gbm_process import *

class PayOffMethods(unittest.TestCase):
    def test_payoff_put(self):
        put_test = PayOffPut(strike=5.0)
        self.assertEqual(put_test(10),0.0)
        self.assertEqual(put_test(2),3.0)
        
    def test_payoff_call(self):
        call_test = PayOffCall(strike=5.0)
        self.assertEqual(call_test(13.0),8.0)
        self.assertEqual(call_test(3),0.0)

    def test_euro_call(self):
        call_test = PayOffCall(strike=5.0)
        euro_call_test = EuropeanOption(call_test, 1)
        self.assertEqual(euro_call_test.PayOff(13.0), 8.0)
        self.assertEqual(euro_call_test.PayOff(3), 0.0)
        self.assertEqual(euro_call_test.Strike,5.0)
        self.assertEqual(euro_call_test.Expiry,1)
        self.assertEqual(euro_call_test.PayOffType,"call")

    def test_euro_put(self):
        put_test = PayOffPut(strike=5.0)
        euro_put_test = EuropeanOption(put_test, 1)
        self.assertEqual(euro_put_test.PayOff(8),0.0)
        self.assertEqual(euro_put_test.PayOff(2),3.0)
        self.assertEqual(euro_put_test.Strike,5.0)
        self.assertEqual(euro_put_test.Expiry,1)
        self.assertEqual(euro_put_test.PayOffType,"put")

class SimulationMethods(unittest.TestCase):

    def test_simulation(self):
        config = SimulationConfig(numberSimus = 100000,
                                  snapshotsims = 100000,
                                  CI = 0.95,
                                  goal = 0.5
                                  )

        test_instrument = BS(spot = 100,
                          sig = 0.5,
                          r = 0.05,
                          option=EuropeanOption(PayOffCall(strike=110), expiry=0.5)
                          )

        mapping = SimMapping(underlying_process = GBM(drift = test_instrument.Q_drift,
                                                      vol = test_instrument.Q_vol
                                                      ),
                            mkt_instrument = test_instrument
                           )

        sim = Simulation(simconfig = config,
                        simmapping = mapping,
                        debug=False
                        )

        sim_status, sim_snapshot = sim.run()
        est_npv = sim_snapshot[1]
        std_error = sim_snapshot[2]/2.0
        upper_bound_est = est_npv + std_error
        lower_bound_est = est_npv - std_error

        self.assertTrue(lower_bound_est <= test_instrument.Analytical_NPV() <= upper_bound_est)

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)


