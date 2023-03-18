import numpy as np


class AndPerceptron:
    def __init__(self):
        self.w1 = 0.5
        self.w2 = 0.5
        self.threshold = 0.7

    def pred(self, x1, x2):
        if self.w1 * x1 + self.w2 * x2 > self.threshold:
            return True
        else:
            return False


class NotAndPerceptron:
    def __init__(self):
        self.p = AndPerceptron()

    def pred(self, x1, x2):
        return not self.p.pred(x1, x2)


class OrPerceptron:
    def __init__(self):
        self.w1 = 0.6
        self.w2 = 0.6
        self.threshold = 0.5

    def pred(self, x1, x2):

        if self.w1 * x1 + self.w2 * x2 > self.threshold:
            return True
        return False


class AndPerceptronNumpy:
    def __init__(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.7

    def pred(self, x1, x2):
        v = np.array([x1, x2])
        return np.sum(v * self.w) + self.b > 0


class XorPerceptronNumpy:
    def __init__(self):
        self.and_gate = AndPerceptronNumpy()
        self.or_gate = OrPerceptron()
        self.not_and_gate = NotAndPerceptron()

    def pred(self, x1, x2):
        r1 = int(self.not_and_gate.pred(x1, x2))
        r2 = int(self.or_gate.pred(x1, x2))

        return self.and_gate.pred(r1, r2)

