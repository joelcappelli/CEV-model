from .payoff import PayOff

class Option:
    def __init__(self, payoff: PayOff, expiry: float):
        self._payoff = payoff
        self._expiry = expiry

    @property
    def Exercise(self):
        pass

    @property
    def PayOffType(self):
        return self._payoff.Type

    @property
    def Expiry(self):
        return self._expiry

    @property
    def Strike(self):
        return self._payoff.Strike

    def PayOff(self, spot: float):
        return self._payoff(spot)

class EuropeanOption(Option):
    def __init__(self, payoff: PayOff, expiry: float):
        self._exercise = expiry
        Option.__init__(self, payoff, expiry)

    @property
    def Exercise(self):
        return self._exercise