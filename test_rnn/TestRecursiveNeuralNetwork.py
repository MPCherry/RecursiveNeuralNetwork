import unittest
import numpy
from rnn import RecursiveNeuralNetwork


class TestRecursiveNeuralNetworkDefaultFunctions(unittest.TestCase):
    def test_half_split(self):
        vector = numpy.zeros((6,))
        left, right = RecursiveNeuralNetwork.half_split(vector)
        self.assertEqual(len(left), 3)
        self.assertEqual(len(right), 3)

        vector = numpy.zeros((5,))
        left, right = RecursiveNeuralNetwork.half_split(vector)
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 3)

    def test_softmax(self):
        vector = [0.5, 0.5]
        self.assertEqual(list(RecursiveNeuralNetwork.softmax(vector)), [0.5, 0.5])

        vector = [0.1, 0.2]
        self.assertEqual(list(RecursiveNeuralNetwork.softmax(vector)), [0.47502081252105999, 0.52497918747894001])

        vector = [0.1, 0.2, 0.3]
        self.assertEqual(list(RecursiveNeuralNetwork.softmax(vector)), [0.30060960535572728,
                                                                        0.33222499353334728,
                                                                        0.36716540111092549])


if __name__ == '__main__':
    unittest.main()
