# This is the main entry point for our research!
# Call from this file
import Data.Generation.Basic as Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    x = pd.read_csv('Data/Storage/test1.csv')
    sns.set_style("darkgrid")
    # hue="size", style="type", linewidth=2,
    sns.scatterplot(data=x, x="x", y="y", palette="flare")
    plt.show()

