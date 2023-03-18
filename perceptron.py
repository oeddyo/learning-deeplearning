

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


class OrPerceptron:
    def __init__(self):
        self.w1 = 0.6
        self.w2 = 0.6
        self.threshold = 0.5
    def pred(self, x1, x2):

        if self.w1 * x1 + self.w2 * x2 > self.threshold:
            return True
        return False
