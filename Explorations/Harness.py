# This is a basic entry point for our research!
# Explore relationship btwn mlp & lin-reg in joining 2 variables together
import Data.Generation.Basic as Basic
import Models.LinReg_Basic as LogReg
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
    print(test_index)
    print(valid_index)


    X_train = df[:valid_index,:-1]
    t_train = df[:valid_index,-1:]

    X_valid = df[valid_index:test_index, :-1]
    t_valid = df[valid_index:test_index, -1:]

    X_test = df[test_index:, :-1]
    t_test = df[test_index:, -1:]



    sns.set_style("darkgrid")
    # MLP.compare_models(X_train, t_train, X_valid, t_valid)
    m1 = MLP.make_model(X_train=X_train,t_train=t_train)
    pred = predict_training(m1, X_valid[:10])

    print(m1.score(X_train, t_train))
    print(m1.score(X_valid, t_valid))

    # diff = pred['prediction'] - pred['X_test1']
    actual_pos = pred['X_test1'] - pred['X_test2']
    # pred.insert(3, column='diff', value=diff)
    # pred.insert(4, column='actual', value=t_valid[:10])
    pred.insert(3, column='actual_pos', value=actual_pos)

    # print(diff)



    f, ax = plt.subplots(figsize=(6, 6))

    sns.scatterplot(data=dfO, x=dfO['x'], y=dfO['y'], marker="s", hue=dfO['x2'], palette="ch:r=-.2,d=.3_r")
    sns.lineplot(data=pred, x="actual_pos", y="prediction", marker="o", c='r')
    # #
    # ax.legend(ncol=2, loc="lower right", frameon=True)

    # sns.pairplot(data=pred, x_vars=["prediction", "X_test1", "X_test2"], y_vars=["diff", "actual"])
    plt.show()

