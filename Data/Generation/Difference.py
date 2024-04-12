# This will have the second feature (x2) correspond with the difference
# between x and y.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def make_linear(start=0, stop=100, nDimensions=3, precision=1):
    """
    The underlying relationship here is linear.
    Given some x value, the y value is identical.

    Note that the underlying distribution is linear
    :param start:
    :param stop:
    :param precision:
    :return:
    """
    offsetPercents = [0.5] * nDimensions

    x0 = np.arange(start, stop, precision)
    y = np.arange(start, stop, precision)

    df = pd.DataFrame()
    df.insert(0, column='y', value=y)

    for i in range(1, nDimensions + 1):
        x_new = (np.random.random((x0.size)) - 0.5) * (stop - start) * offsetPercents[i-1] + start
        x0 = x0 + x_new
        df.insert(i, column=f'x{i}', value=x_new)

    df.insert(i+1, column='x0', value=x0)

    return df

def make_difference(start=0, stop=100, nDimensions=3, precision=0.1):
    """
    The underlying relationship here is linear.
    Given some x value, the y value is identical.

    Note that the underlying distribution is linear
    :param start:
    :param stop:
    :param precision:
    :return:
    """
    offsetPercents = [0.1] * nDimensions

    x0 = np.arange(start, stop, precision)
    y = np.arange(start, stop, precision)

    df = pd.DataFrame()
    df.insert(0, column='y', value=y)
    for i in range(1, nDimensions + 1):
        # x_new = (np.random.random((x0.size)) - 0.5) * (stop - start) * offsetPercents[i-1] + start
        x_new, feature = sin_difference(x0, size=offsetPercents[i-1])
        x0 = x0 + x_new
        df.insert(i, column=f'x{i}', value=feature)

    df.insert(i+1, column='x0', value=x0)

    return df


def linear_difference(X, size=0.2):
    min = X.min()
    max = X.max()
    return (np.random.random((X.size)) - 0.5) * (max - min) * size + min
    # return x_diff

def sin_difference(X, size=0.2):
    period = np.linspace(0, math.pi * 2, X.shape[0])
    np.random.shuffle(period)
    x_diff = np.sin(period) * 10
    print(x_diff)

    # return (np.random.random((X.size)) - 0.5) * (max - min) * size + min
    return x_diff, period

def multiply_difference():
    multipliers = np.random.randint(0, 100)


def calc_dist(df):
    dist = df['x0'] - df['y']
    df.insert(df.shape[1], column='diff', value=dist)


if __name__ == "__main__":
    nDim = 1
    y = make_difference(nDimensions=nDim)
    calc_dist(y)
    print(y)


    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(data=y, x="x0", y="y", hue="diff", palette="flare", ax=ax[0])

    diff_caused = y['y'] - y['x0']
    sns.scatterplot(data=y, x="x1", y=diff_caused, hue="diff", palette="flare", ax=ax[1])

    plt.show()
    y.to_csv(f'Data/Storage/difference_{nDim}.csv', index=False)
