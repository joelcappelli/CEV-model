
class PayOff:
    def __init__(self, strike: float):
        self._strike = strike

    @property
    def Strike(self):
        return self._strike

    def __call__(self, spot: float):
        pass

class PayOffCall(PayOff):
    def __init__(self, strike: float):
        self._type = 'call'
        PayOff.__init__(self, strike)

    @property
    def Type(self):
        return self._type

    def __call__(self, spot: float):
        return max(spot - self._strike, 0.0)

class PayOffPut(PayOff):
    def __init__(self, strike: float):
        self._type = 'put'
        PayOff.__init__(self, strike)

    @property
    def Type(self):
        return self._type

    def __call__(self, spot: float):
        return max(self._strike - spot, 0.0)