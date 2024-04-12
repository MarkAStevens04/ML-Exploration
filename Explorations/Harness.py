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


def predict_training(model, X_train):
    df2 = pd.DataFrame(index=range(X_train.shape[0]))

    pred = model.predict(X_train)
    df2.insert(0, column="X0", value=X_train[:, -1:])
    for i in range(X_train.shape[1] - 1):
        df2.insert(i + 1, column=f"X{i + 1}", value=X_train[:, i])
    df2.insert(i + 2, column='prediction', value=pred)

    return df2

#
# def reconstruct(X_train):
#     actual_start = X_train[:, -1:]
#     for i in range(X_train.shape[1] - 1):
#         actual_start = actual_start - X_train[:, i:i+1]
#     return actual_start




if __name__ == "__main__":
    nDim = 1
    dfO = pd.read_csv(f'Data/Storage/difference_{nDim}.csv', header=0)
    print(dfO)

    df = dfO.to_numpy()
    np.random.shuffle(df)
    # hue="size", style="type", linewidth=2,
    test_size = 0.10
    valid_size = 0.10
    test_index = df.shape[0] - int(df.shape[0] * test_size)
    valid_index = test_index - int(df.shape[0] * test_size)

    X_train = df[:valid_index,1:-1]
    t_train = df[:valid_index,:1]

    X_valid = df[valid_index:test_index,1:-1]
    t_valid = df[valid_index:test_index, :1]

    X_test = df[test_index:, 1:-1]
    t_test = df[test_index:, :1]


    sns.set_style("darkgrid")

    # MLP.compare_models(X_train, t_train, X_valid, t_valid)

    X_sample = X_valid[:10]
    t_sample = t_valid[:10]


    m1 = MLP.make_model(X_train=X_train,t_train=t_train)
    m2 = LinReg.make_model(X_train=X_train,t_train=t_train)

    predMLP = predict_training(m1, X_sample)
    predLin = predict_training(m2, X_sample)

    print(m1.score(X_train, t_train))
    print(m1.score(X_valid, t_valid))

    print(m2.score(X_train, t_train))
    print(m2.score(X_valid, t_valid))

    # actual_pos = reconstruct(X_sample)
    # predMLP.insert(predMLP.shape[1], column='actual_pos', value=actual_pos)

    # actual_pos = reconstruct(X_sample)
    # predLin.insert(predLin.shape[1], column='actual_pos', value=actual_pos)

    f, ax = plt.subplots(2, 2, figsize=(12, 12))

    print(predMLP)
    # diff_caused = t_sample.ravel() - predMLP['X0']
    diff2 = dfO['y'] - dfO['x0']
    diff_found = t_sample.ravel() - predMLP['X0']
    # diff_found = t_sample.ravel() - predMLP['prediction'] +


    sns.scatterplot(data=dfO, x="x0", y="y", marker="s", hue="diff", palette="ch:r=-.2,d=.3_r", ax=ax[0, 1])
    sns.lineplot(x=[0, 100], y=[0, 100], c='black', ax=ax[0, 1], linewidth=2, label="actual")
    sns.lineplot(data=predMLP, x="prediction", y=t_sample.ravel(), marker="o", c='r', ax=ax[0, 1])

    sns.scatterplot(data=dfO, x="x1", y=diff2, marker="s", hue="diff", palette="ch:r=-.2,d=.3_r", ax=ax[1, 1])
    # sns.lineplot(x=[0, 100], y=[0, 100], c='black', ax=ax[1, 1], linewidth=2, label="actual")
    sns.lineplot(data=predMLP, x="X1", y=diff_found, marker="o", c='r', ax=ax[1, 1])


    sns.scatterplot(data=dfO, x="x0", y="y", marker="s", hue="diff", palette="ch:r=-.2,d=.3_r", ax=ax[0, 0])
    sns.lineplot(x=[0, 100], y=[0, 100], c='black', ax=ax[0, 0], linewidth=2, label="actual")
    sns.lineplot(data=predLin, x="prediction", y=t_sample.ravel(), marker="o", c='b', ax=ax[0, 0])

    ax[0, 0].set(title="Linear Regression", xlabel="prediction", ylabel="target")
    ax[0, 1].set(title="MLP", xlabel="prediction", ylabel="target")

    plt.show()

