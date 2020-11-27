import numpy as np

class OneHotEncoder:
    def __init__(self, y):
        yEncoded = np.zeros((y.max() + 1, y.size))
        yEncoded[y, np.arange(y.size)] = 1
        self.yEncoded = yEncoded