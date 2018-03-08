class KNearestNeighbor(object):

    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            pass
