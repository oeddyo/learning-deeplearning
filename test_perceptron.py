import unittest
from perceptron import AndPerceptron
from perceptron import NotAndPerceptron
from perceptron import AndPerceptronNumpy
from perceptron import OrPerceptron
from perceptron import XorPerceptronNumpy


class TestPerceptron(unittest.TestCase):
    def test_and_gate(self):
        p = AndPerceptron()
        self.assertEqual(p.pred(1, 1), True)
        self.assertEqual(p.pred(1, 0), False)
        self.assertEqual(p.pred(0, 1), False)
        self.assertEqual(p.pred(0, 0), False)

    def test_not_and(self):
        p = NotAndPerceptron()
        self.assertEqual(p.pred(1, 1), False)
        self.assertEqual(p.pred(1, 0), True)
        self.assertEqual(p.pred(0, 1), True)
        self.assertEqual(p.pred(0, 0), True)

    def test_or_gate(self):
        p = OrPerceptron()

        self.assertEqual(p.pred(1, 1), True)
        self.assertEqual(p.pred(1, 0), True)
        self.assertEqual(p.pred(0, 1), True)
        self.assertEqual(p.pred(0, 0), False)

    def test_and_gate_numpy(self):
        p = AndPerceptronNumpy()
        self.assertEqual(p.pred(1, 1), True)
        self.assertEqual(p.pred(1, 0), False)
        self.assertEqual(p.pred(0, 1), False)
        self.assertEqual(p.pred(0, 0), False)

    def test_xor_gate(self):
        p = XorPerceptronNumpy()
        self.assertEqual(p.pred(1, 1), False)
        self.assertEqual(p.pred(1, 0), True)
        self.assertEqual(p.pred(0, 1), True)
        self.assertEqual(p.pred(0, 0), False)
