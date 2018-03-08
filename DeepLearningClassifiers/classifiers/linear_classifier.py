import numpy as np


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0... K-1 where K is number of classes
        if None == self.W:
            # lazily initialize W
            self.W = 0.001 * np.random.rand(dim, num_classes)
