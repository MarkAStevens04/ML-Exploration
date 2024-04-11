# This will have the second feature (x2) correspond with the difference
# between x and y.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_linear(start=0, stop=100, precision=0.1):
    """
    The underlying relationship here is linear.
    Given some x value, the y value is identical.
    :param start:
    :param stop:
    :param precision:
    :return:
    """
    offsetPercent = 0.2
    x = np.arange(start, stop, precision)
    y = np.arange(start, stop, precision)


    x2 = (np.random.random((x.size)) - 0.5) * (stop - start) * offsetPercent + start
    x = x + x2

    df = pd.DataFrame()
    df.insert(0, column='x', value=x)
    df.insert(1, column='x2', value=x2)
    df.insert(2, column='y', value=y)
    return df


if __name__ == "__main__":
    y = make_linear()
    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    sns.scatterplot(data=y, x="x", y="y", hue="x2", palette="flare")

    plt.show()
    y.to_csv('Data/Storage/difference.csv', index=False)
