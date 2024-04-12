# This will have the second feature (x2) correspond with the difference
# between x and y.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

SCALE = 10
np.random.seed(42)


def make_linear(start=0, stop=100, nDimensions=3, precision=0.01):
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

def make_difference(start=0, stop=100, nDimensions=3, precision=0.01):
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



    # original_period = np.linspace(0, math.pi * 2, x0.shape[0])
    #
    # previous_value, previous_index = sin_difference(original_period, size=offsetPercents[0])
    # df.insert(1, column=f'x3', value=original_period)
    # new_value, new_index = sin_difference(previous_value, size=offsetPercents[0])
    # # df.insert(2, column=f'x2', value=new_value)
    # df.insert(2, column=f'x2', value=new_index)
    # newer_value, newer_index = sin_difference(new_value, size=offsetPercents[0])
    # df.insert(3, column='x1', value=newer_value)
    # x0 = x0 + newer_value
    #
    # # df.insert(2, column=f'x2', value=previous_value)
    #
    # df.insert(4, column=f'x0', value=x0)






    previous_period = np.linspace(0, math.pi * 2, x0.shape[0])
    df.insert(1, column=f'x1', value=previous_period)
    for i in range(1, nDimensions + 1):
        new_value, new_index = sin_difference(previous_period, size=offsetPercents[0])
        df.insert(1 + i, column=f'x{i+1}', value=new_index)
        previous_period = new_value

    x0 = x0 + new_value
    # df.insert(i+2, column=f'x4', value=p)

    df.insert(i+2, column=f'x0', value=x0)

    return df


def linear_difference(X, size=0.2):
    min = X.min()
    max = X.max()
    return (np.random.random((X.size)) - 0.5) * (max - min) * size + min
    # return x_diff

def sin_difference(X, size=0.2):
    period = np.linspace(0, math.pi * 2, X.shape[0])
    np.random.shuffle(period)
    x_diff = np.sin(period + X) * SCALE
    print(x_diff)

    # return (np.random.random((X.size)) - 0.5) * (max - min) * size + min
    return x_diff, period

def multiply_difference():
    multipliers = np.random.randint(0, 100)


def calc_dist(df):
    dist = df['x0'] - df['y']
    df.insert(df.shape[1], column='diff', value=dist)

# def flatten_to_2pi(s):
#     s = s * -1
#     s = s - math.pi
#     # while s < 0:
#     #     s += math.pi
#     # while s > 2 * math.pi:
#     #     s -= math.pi
#     print(s)
#     return s

if __name__ == "__main__":
    nDim = 10
    y = make_difference(nDimensions=nDim)
    calc_dist(y)
    print(y)


    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    f, ax = plt.subplots(3, y.shape[1] // 2, figsize=(12, 6))
    sns.scatterplot(data=y, x="x0", y="y", hue="diff", palette="flare", ax=ax[0, 0])

    # diff_caused = y['y'] - y['x0']
    # # diff_2 = y['x1']
    # diff_2 = np.sin(y['x1'])
    # diff_3 = np.sin(y['x2'])
    # # diff_4 = np.sin(y['x3'])
    # temp = np.arcsin(y['diff']/10) + y['x3']
    # # diff_4 = np.apply_along_axis(flatten_to_2pi, 1, temp)
    # t2 = np.vectorize(flatten_to_2pi)
    # diff_4 = t2(temp)
    #
    # diffs = [diff_2, diff_3, diff_4]
    # curr_diff = np.sin(y['x1'])
    # for i in range(nDim + 1):
    #     a = sns.scatterplot(data=y, x=f"x{i+1}", y=diffs[i], hue="diff", palette="flare", ax=ax[1 + i%2, i//2])
    #     ax[1 + i%2, i//2].legend([],[], frameon=False)
    #     ax[1 + i % 2, i // 2].set(ylabel=f'x{i+1}')
    #     if i < nDim + 1:
    #         curr_diff = np.sin(y[f'x{i+1}'] + curr_diff/SCALE)

    for i in range(nDim + 1):
        a = sns.scatterplot(data=y, x=f"x{i+1}", y=np.sin(y[f'x{i+1}']), hue="diff", palette="flare", ax=ax[1 + i%2, i//2])
        ax[1 + i%2, i//2].legend([],[], frameon=False)
        ax[1 + i % 2, i // 2].set(ylabel=f'x{i+1}')

    print(nDim)
    y.to_csv(f'Data/Storage/difference_{nDim}.csv', index=False)
    plt.show()

