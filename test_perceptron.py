import unittest
from perceptron import AndPerceptron


class TestPerceptron(unittest.TestCase):
    def test_and_gate(self):
        p = AndPerceptron()
        self.assertEqual(p.pred(1, 1), True)
        self.assertEqual(p.pred(1, 0), False)
        self.assertEqual(p.pred(0, 1), False)
        self.assertEqual(p.pred(0, 0), False)


