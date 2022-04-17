import math

class Parameter:
    def __init__(self, init_val: float):
        self._value = init_val

    def Integral(self,t1: float, t2: float):
        pass

    def IntegralSqu(self, t1: float, t2: float):
        pass

    def Mean(self, t1: float, t2: float):
        total = self.Integral(t1,t2)
        return total/(t2-t1)

    def RootMeanSqu(self, t1: float, t2: float):
        total = self.IntegralSqu(t1,t2)
        return math.sqrt(total/(t2-t1))

    def __call__(self):
        return self._value

class Constant(Parameter):
    def __init__(self, init_val: float):
        Parameter.__init__(self,init_val)

    def Integral(self, t1: float, t2: float):
        return self._value*(t2-t1)

    def IntegralSqu(self, t1: float, t2: float):
        return self._value*self._value*(t2-t1)