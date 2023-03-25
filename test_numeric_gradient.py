import unittest

import os
import sys
import numpy as np
from common.gradient import numerical_gradient, numerical_gradient_faster



sys.path.append(os.pardir)


class TestPerceptron(unittest.TestCase):
    def testtest(self):

        def f(x):
            return np.sum(x * 2)

        a = np.array([1,2])
        print("normal ", numerical_gradient(f, a))
        print("faster ", numerical_gradient_faster(f, a))

