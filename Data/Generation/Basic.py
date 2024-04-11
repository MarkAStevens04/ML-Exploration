# Basic data creation methods!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def create_file(dir):
    x = open(dir, 'x+')

def make_random(start=0, stop=1, precision=0.01):
    n = int(1 / precision)
    # x = pd.DataFrame(np.random.randint(low=start, high=stop, size=(n,2)), columns=['x', 'y'])
    x = pd.DataFrame(np.random.random((n,2)), columns=['x', 'y'])
    print(x)
    return x

def make_linear(start=0, stop=1, precision=0.01):
    """
    The underlying relationship here is linear.
    Given some x value, the y value is identical.
    :param start:
    :param stop:
    :param precision:
    :return:
    """
    x = np.arange(start, stop, precision)
    x2 = np.arange(start, stop, precision)
    y = np.arange(start, stop, precision)
    df = pd.DataFrame()
    df.insert(0, column='x', value=x)
    df.insert(1, column='x2', value=x2)
    df.insert(2, column='y', value=y)
    return df


if __name__ == "__main__":
    x = make_random()
    y = make_linear()

    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    sns.scatterplot(data=x, x="x", y="y", palette="flare")

    y.to_csv('Data/Storage/test1.csv', index=False)
    plt.show()

