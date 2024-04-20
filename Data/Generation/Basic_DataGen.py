import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Data.Generation.Abstract_DataGen import AbstractDataGen

class BasicGen(AbstractDataGen):
    def __init__(self, start=0, stop=1, n_points=100):
        self.data = None
        self.params = None
        self.start = start
        self.stop = stop
        self.n_points = int(n_points)

    def generate(self):
        """
        The underlying distribution here is linear.
        Given some x value, the y value is identical.
        :param start:
        :param stop:
        :param precision:
        :return:
        """
        
        x = np.arange(self.start, self.stop, self.precision)
        x2 = np.arange(self.start, self.stop, self.precision)
        y = np.arange(self.start, self.stop, self.precision)
        df = pd.DataFrame()
        df.insert(0, column='x', value=x)
        df.insert(1, column='x2', value=x2)
        df.insert(2, column='y', value=y)
        return df

    def _make_random(self):
        x = pd.DataFrame(np.random.randint(low=self.start, high=self.stop, size=(self.n_points,2)), columns=['x', 'y'])
        return x
