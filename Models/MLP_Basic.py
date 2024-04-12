from sklearn.neural_network import MLPRegressor
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def make_model(X_train, t_train):
    model = MLPRegressor(hidden_layer_sizes=(7, 7), max_iter=10000)
    t_train = t_train.ravel()
    model.fit(X=X_train, y=t_train)

    return model


def compare_models(X_train, t_train, X_valid, t_valid):
    accuracies = []

    t_train = t_train.ravel()
    t_valid = t_valid.ravel()

    sizes = [5, 7, 9, 13, 15]
    num_layers = [3]

    sizeGraph = True

    for s in sizes:
        for n in num_layers:
            sizes = [s] * n
            mlp = MLPRegressor(hidden_layer_sizes=sizes, activation='relu')
            for i in range(1000):
                mlp.partial_fit(X_train, t_train)
                if i > 100 and i % 10 == 0:
                    a = mlp.score(X_train, t_train)
                    v = mlp.score(X_valid, t_valid)
                    # accuracies.append([i, a, "training", f"{n}x{s}"])
                    # accuracies.append([i, v, "validation", f"{n}x{s}"])
                    if sizeGraph:
                        accuracies.append([i, a, "training", s])
                        accuracies.append([i, v, "validation", s])
                    else:
                        accuracies.append([i, a, "training", n])
                        accuracies.append([i, v, "validation", n])

    # ideal is 2x7
    # n-iter is 100

    if sizeGraph:
        acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    else:
        acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "depth"])

    sns.set_style("darkgrid")
    palette = sns.cubehelix_palette(light=.8, n_colors=4)
    # flare is red & orange, crest is green & blue
    if sizeGraph:
        sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="size", style="type", linewidth=2, palette="flare")
    else:
        sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="depth", style="type", linewidth=2, palette="crest")

    plt.show()