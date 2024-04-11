# This will have the second feature (x2) correspond with the difference
# between x and y.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def calc_dist(df):
    dist = df['x0'] - df['y']
    df.insert(df.shape[1], column='diff', value=dist)


if __name__ == "__main__":
    nDim = 30
    y = make_linear(nDimensions=nDim)
    calc_dist(y)
    print(y)
    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    sns.scatterplot(data=y, x="x0", y="y", hue="diff", palette="flare")

    plt.show()
    y.to_csv(f'Data/Storage/difference_{nDim}.csv', index=False)
