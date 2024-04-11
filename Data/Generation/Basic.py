# Basic data creation methods!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def create_file(dir):
    x = open(dir, 'x+')

def make_random(start=0, stop=100, precision=1):
    n = int(1 / precision * 100)
    x = pd.DataFrame(np.random.randint(low=start, high=stop, size=(n,2)), columns=['x', 'y'])
    return x

def make_linear(start=0, stop=100, precision=0.1):
    x = np.arange(start, stop, precision)
    y = np.arange(start, stop, precision)
    df = pd.DataFrame()
    df.insert(0, column='x', value=x)
    df.insert(1, column='y', value=y)
    return df


if __name__ == "__main__":
    x = make_random()
    y = make_linear()

    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    sns.scatterplot(data=x, x="x", y="y", palette="flare")

    y.to_csv('Data/Storage/test1.csv')
    plt.show()

