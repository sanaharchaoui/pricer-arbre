from typing import Callable


# dérivée d'une fonction ; la fonction est passée en paramètre
class OneDimDerivative(object):

    def __init__(self, function: Callable[[object, float], float], other_parameters: object,
                 shift: float = 1):
        self.f: Callable[[object, float], float] = function
        self.param: object = other_parameters
        self.shift: float = shift

    # différence centrée : (f(x + shift) - f(x - shift)) / (2 shift)
    def first(self, x: float) -> float:
        return (self.f(self.param, x + self.shift) - self.f(self.param, x - self.shift)) / (2 * self.shift)
