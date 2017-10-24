import numpy


def half_split(vector):
    length = len(vector)
    return vector[:length // 2], vector[length // 2:]


# Taken from this Stack Overflow question
# https://stackoverflow.com/q/34968722
def softmax(vector):
    e_vector = numpy.exp(vector - numpy.max(vector))
    return e_vector / e_vector.sum()


class RecursiveNeuralNetwork(object):
    def __init__(self, vector_len, classes=1, split_fn=half_split, node_fn=numpy.tanh, classification_fn=softmax):
        """
        vector_len: The length of the input vectors that will be classified

        classes:  The number of classification classes to output

        split_fn: The function used to recursively branch the recursion tree
            Must take a list and return a two-tuple of lists

        node_fn:  The function used to normalize a tree node
            Must take a list and return a list of the same length
            Applied to the net input of each node in the tree, excluding the output layer

        classification_fn: The function used to produce a classification output
            Must take a list and return a list of the same length
            Applied to the net input of the output layer
        """
        self.classes = classes
        self.split_fn = split_fn
        self.node_fn = node_fn
        self.classification_fn = classification_fn

        self.m_classify = numpy.ones((classes, vector_len))
        self.m_left = numpy.ones((vector_len, vector_len))
        self.m_right = numpy.ones((vector_len, vector_len))

    def combine(self, left, right):
        combined = numpy.dot(self.m_left, left) + numpy.dot(self.m_right, right)
        return self.node_fn(combined)

    def recursive_evaluate(self, vector_list):
        if len(vector_list) == 1:
            return vector_list[0]
        else:
            left, right = self.split_fn(vector_list)
            return self.combine(self.recursive_evaluate(left), self.recursive_evaluate(right))

    def classify(self, vector_list):
        output_vector = self.recursive_evaluate(vector_list)
        return self.classification_fn(numpy.dot(self.m_classify, output_vector))