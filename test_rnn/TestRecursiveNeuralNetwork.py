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


def node_fn_0(vector):
    return vector


def node_fn_1(vector):
    new_vector = []
    for num in vector:
        new_vector.append(num*2)
    return new_vector


def split_fn_0(vector_list):
    return vector_list[:-1], vector_list[-1:]


class TestRecursiveNeuralNetworkPrediction(unittest.TestCase):
    def test_combine(self):
        rnn = RecursiveNeuralNetwork.RecursiveNeuralNetwork(3, 1, node_fn=node_fn_0)
        rnn.m_left = [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]
        rnn.m_right = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]

        combination = rnn.combine([1, 1, 1], [1, 1, 1])
        self.assertEquals(list(combination), [2, 2, 2])

        combination = rnn.combine([1, 1, 1], [2, 2, 2])
        self.assertEquals(list(combination), [3, 3, 3])

        combination = rnn.combine([1, 2, 3], [5, 2, 7])
        self.assertEquals(list(combination), [6, 4, 10])

        rnn.m_left = [[1, 2, 3],
                      [1, 1, 1],
                      [0, 0, 0]]
        rnn.m_right = [[1, 2, 1],
                       [1, 0, 1],
                       [1, 0, 3]]

        combination = rnn.combine([1, 1, 1], [1, 1, 1])
        self.assertEquals(list(combination), [10, 5, 4])

        combination = rnn.combine([1, 1, 1], [2, 2, 2])
        self.assertEquals(list(combination), [14, 7, 8])

        combination = rnn.combine([1, 2, 3], [5, 2, 7])
        self.assertEquals(list(combination), [30, 18, 26])

        rnn.node_fn = node_fn_1

        combination = rnn.combine([1, 1, 1], [1, 1, 1])
        self.assertEquals(list(combination), [20, 10, 8])

        combination = rnn.combine([1, 1, 1], [2, 2, 2])
        self.assertEquals(list(combination), [28, 14, 16])

        combination = rnn.combine([1, 2, 3], [5, 2, 7])
        self.assertEquals(list(combination), [60, 36, 52])

    def test_recursive_evaluate(self):
        rnn = RecursiveNeuralNetwork.RecursiveNeuralNetwork(1, 1, node_fn=node_fn_0)  # use default half split
        rnn.m_left = [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]
        rnn.m_right = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]

        # Test base case
        evaluation = rnn.recursive_evaluate([[1, 1, 1]])
        self.assertEquals(list(evaluation), [1, 1, 1])

        rnn.m_left = [[2]]
        rnn.m_right = [[2]]

        evaluation = rnn.recursive_evaluate([[1], [1], [1], [1]])
        self.assertEquals(list(evaluation), [16])

        rnn.split_fn = split_fn_0

        evaluation = rnn.recursive_evaluate([[1], [1], [1], [1]])
        self.assertEquals(list(evaluation), [22])

    def test_predict(self):
        rnn = RecursiveNeuralNetwork.RecursiveNeuralNetwork(2, 2, node_fn=node_fn_0)  # use default half split, softmax
        rnn.m_left = [[1, 0], [0, 1]]
        rnn.m_right = [[1, 0], [0, 1]]
        rnn.m_classify = [[1, 0], [0, 1]]

        prediction = rnn.predict([[1, 1], [1, 1], [1, 1], [1, 1]])
        self.assertEquals(list(prediction), [0.5, 0.5])

        prediction = rnn.predict([[1, 0], [1, 0], [1, 0], [1, 0]])
        self.assertTrue(prediction[0] - prediction[1] > 0.9)

        rnn.classification_fn = node_fn_0

        prediction = rnn.predict([[1, 1], [1, 1], [1, 1], [1, 1]])
        self.assertEquals(list(prediction), [4, 4])


if __name__ == '__main__':
    unittest.main()
