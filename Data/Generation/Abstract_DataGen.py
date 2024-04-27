import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AbstractDataGen:
    def __init__(self):
        self.data = None
        self.params = None

    def generate(self):
        self.data = None
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def split(self, train=0.8, test=0.1, valid=0.1):
        train = train
        valid = test
        test = valid

        test_size = int(len(self.data) * test)
        valid_size = int(len(self.data) * valid)

        test_index = len(self.data) - test_size
        valid_index = test_index - valid_size

        X_train = np.array(self.data[['x', 'x2']][:valid_index])
        X_valid = self.data[['x', 'x2']][valid_index:test_index]
        X_test = self.data[['x', 'x2']][test_index:]

        t_train = np.array(self.data['y'][:valid_index])
        t_valid = self.data['y'][valid_index:test_index]
        t_valid = self.data['y'][test_index:]

        return X_train, X_valid, X_test, t_train, t_valid, t_valid
