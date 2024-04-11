# This is a basic entry point for our research!
# Explore relationship btwn mlp & lin-reg in joining 2 variables together
import Data.Generation.Basic as Basic
import Models.LinReg_Basic as LinReg
import Models.MLP_Basic as MLP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


def predict_range(model, precision=0.1):
    n = int(1/precision) + 1
    df2 = pd.DataFrame(index=range(n))
    # X_test1 = np.arange(start=0, stop=1, step=precision)
    # X_test2 = np.arange(start=0, stop=1, step=precision)
    X_test = np.linspace(start=[0, 0], stop=[1, 1], num=n)
    # X_test.insert(0, column='x1', value=X_test1)
    # X_test.insert(1, column='x2', value=X_test2)

    pred = model.predict(X_test)
    df2.insert(0, column='X_test1', value=X_test[:, :1])
    df2.insert(1, column='X_test2', value=X_test[:, 1:])
    df2.insert(2, column='prediction', value=pred)
    return df2

def predict_training(model, X_train):
    df2 = pd.DataFrame(index=range(X_train.shape[0]))

    pred = model.predict(X_train)
    df2.insert(0, column='X_test1', value=X_train[:, :1])
    df2.insert(1, column='X_test2', value=X_train[:, 1:])
    df2.insert(2, column='prediction', value=pred)
    return df2




if __name__ == "__main__":
    dfO = pd.read_csv('Data/Storage/difference.csv', header=0)

    df = dfO.to_numpy()
    np.random.shuffle(df)
    # hue="size", style="type", linewidth=2,
    test_size = 0.10
    valid_size = 0.10
    test_index = df.shape[0] - int(df.shape[0] * test_size)
    valid_index = test_index - int(df.shape[0] * test_size)

    X_train = df[:valid_index,:-1]
    t_train = df[:valid_index,-1:]

    X_valid = df[valid_index:test_index, :-1]
    t_valid = df[valid_index:test_index, -1:]

    X_test = df[test_index:, :-1]
    t_test = df[test_index:, -1:]



    sns.set_style("darkgrid")
    # MLP.compare_models(X_train, t_train, X_valid, t_valid)
    m1 = MLP.make_model(X_train=X_train,t_train=t_train)
    m2 = LinReg.make_model(X_train=X_train,t_train=t_train)

    predMLP = predict_training(m1, X_valid[:10])
    predLin = predict_training(m2, X_valid[:10])

    print(m1.score(X_train, t_train))
    print(m1.score(X_valid, t_valid))

    print(m2.score(X_train, t_train))
    print(m2.score(X_valid, t_valid))

    actual_pos = predMLP['X_test1'] - predMLP['X_test2']
    predMLP.insert(3, column='actual_pos', value=actual_pos)

    actual_pos = predLin['X_test1'] - predLin['X_test2']
    predLin.insert(3, column='actual_pos', value=actual_pos)


    f, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(data=dfO, x="x", y="y", marker="s", hue="x2", palette="ch:r=-.2,d=.3_r", ax=ax[1])
    sns.lineplot(data=predMLP, x="actual_pos", y="prediction", marker="o", c='r', ax=ax[1])

    sns.scatterplot(data=dfO, x="x", y="y", marker="s", hue="x2", palette="ch:r=-.2,d=.3_r", ax=ax[0])
    sns.lineplot(data=predLin, x="actual_pos", y="prediction", marker="o", c='b', ax=ax[0])

    ax[0].set(title="Linear Regression")
    ax[1].set(title="MLP")

    # #
    # ax.legend(ncol=2, loc="lower right", frameon=True)

    # sns.pairplot(data=pred, x_vars=["prediction", "X_test1", "X_test2"], y_vars=["diff", "actual"])
    plt.show()

