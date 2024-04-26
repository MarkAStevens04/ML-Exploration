import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Data.Generation.Abstract_DataGen import AbstractDataGen

class BasicGen(AbstractDataGen):
    def __init__(self, start=0, stop=1, n_points=100):
        super().__init__()
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
        x = np.arange(start=self.start, stop=self.stop, step=1/self.n_points)
        x2 = np.arange(start=self.start, stop=self.stop, step=1/self.n_points)
        y = np.arange(start=self.start, stop=self.stop, step=1/self.n_points)
        df = pd.DataFrame()
        df.insert(0, column='x', value=x)
        df.insert(1, column='x2', value=x2)
        df.insert(2, column='y', value=y)

        self.data = df
        return df

    def _make_random(self):
        x = pd.DataFrame(np.random.randint(low=self.start, high=self.stop, size=(self.n_points,2)), columns=['x', 'y'])
        return x
