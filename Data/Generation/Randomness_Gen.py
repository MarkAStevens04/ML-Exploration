import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Data.Generation.Abstract_DataGen import AbstractDataGen

class RandomGen(AbstractDataGen):
    def __init__(self, start=0, stop=1, n_points=100):
        super().__init__()
        self.data = None
        self.params = None
        self.start = start
        self.stop = stop
        self.n_points = int(n_points)

    def generate(self, randomness=0.1):
        """
        The underlying distribution here is linear.
        Given some x value, the y value is identical.
        :param start:
        :param stop:
        :param precision:
        :return:
        """
        x = np.arange(start=self.start, stop=self.stop, step=1/self.n_points)
        x_modify = np.random.random((self.n_points, ))
        x = x - (x_modify * 0.5)

        y = np.arange(start=self.start, stop=self.stop, step=1/self.n_points)
        df = pd.DataFrame()
        df.insert(0, column='x', value=x)
        # df.insert(1, column='x2', value=x2)
        df.insert(1, column='y', value=y)

        self.data = df
        self.display_data()
        return df

    def _make_random(self):
        x = pd.DataFrame(np.random.randint(low=self.start, high=self.stop, size=(self.n_points,2)), columns=['x', 'y'])
        return x

    def display_data(self):
        sns.set_style("darkgrid")
        # hue="size", style="type", linewidth=2,
        sns.scatterplot(data=self.data, x="x", y="y", palette="flare")

        # y.to_csv('Data/Storage/test1.csv', index=False)
        plt.show()
        print(f'attempting to display...')
