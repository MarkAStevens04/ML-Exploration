from sklearn.neural_network import MLPRegressor
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_model(X_train, t_train):
    model = MLPRegressor(hidden_layer_sizes=(15, 15, 15), max_iter=10000, activation='relu')
    t_train = t_train.ravel()
    model.fit(X=X_train, y=t_train)

    return model


def compare_models(X_train, t_train, X_valid, t_valid):
    accuracies = []
    predictions = []
    differences = []

    t_train = t_train.ravel()
    t_valid = t_valid.ravel()

    sizes = [5, 10, 15, 20]
    num_layers = [3]

    sizeGraph = True
    j = 0

    for s in sizes:
        for n in num_layers:
            sizes = [s] * n
            mlp = MLPRegressor(hidden_layer_sizes=sizes, activation='relu')
            j = -1
            for i in range(10000):
                mlp.partial_fit(X_train, t_train)
                if i > 100 and i % 10 == 0:
                    j += 1
                    if j >= len(predictions):
                        predictions.append([])
                        differences.append([])
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

                    pred = predict_training(mlp, X_valid)
                    diff_found = pred - X_valid[:, 1]
                    predictions[j].append(diff_found)
                    # dist = diff_found - np.sin(X_valid[:, 0])
                    dist = t_valid - pred
                    # differences.append(t_valid - pred)
                    differences[j].append(dist)






    # ideal is 2x7
    # n-iter is 100

    if sizeGraph:
        acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    else:
        acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "depth"])

    sns.set_style("darkgrid")
    palette = sns.cubehelix_palette(light=.8, n_colors=4)
    # flare is red & orange, crest is green & blue


    # if sizeGraph:
    #     sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="size", style="type", linewidth=2, palette="flare")
    # else:
    #     sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="depth", style="type", linewidth=2, palette="crest")

    pred = predict_training(mlp, X_valid)
    diff_found = pred - X_valid[:, 1]

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].set(title="5, 5, 5")
    ax[0, 1].set(title="10, 10, 10")
    ax[1, 0].set(title="15, 15, 15")
    ax[1, 1].set(title="20, 20, 20")

    norm = plt.Normalize(vmin=-5, vmax=5)
    ani = animation.FuncAnimation(fig, animate, fargs=[ax, X_valid, predictions, differences, norm], frames=len(predictions), interval=1, repeat=True)
    ani.save('animation_drawing.gif', writer='Pillow', fps=240)
    plt.show()



def animate(i, ax, X_valid, predictions, differences, norm):
    # plt.clf()
    ax[1, 1].cla()
    ax[0, 1].cla()
    ax[1, 0].cla()
    ax[0, 0].cla()
    ax[0, 0].set(title="5, 5, 5")
    ax[0, 1].set(title="10, 10, 10")
    ax[1, 0].set(title="15, 15, 15")
    ax[1, 1].set(title="20, 20, 20")
    ax[0, 0].set_ylim([-12, 12])
    ax[0, 1].set_ylim([-12, 12])
    ax[1, 0].set_ylim([-12, 12])
    ax[1, 1].set_ylim([-12, 12])

    sns.scatterplot(x=X_valid[:, 0], y=predictions[i][0], marker="o", c='r', hue=differences[i][0], ax=ax[0, 0], palette="icefire", hue_norm=norm)
    sns.scatterplot(x=X_valid[:, 0], y=predictions[i][1], marker="o", c='r', hue=differences[i][1], ax=ax[0, 1], palette="icefire", hue_norm=norm)
    sns.scatterplot(x=X_valid[:, 0], y=predictions[i][2], marker="o", c='r', hue=differences[i][2], ax=ax[1, 0], palette="icefire", hue_norm=norm)
    sns.scatterplot(x=X_valid[:, 0], y=predictions[i][3], marker="o", c='r', hue=differences[i][3], ax=ax[1, 1], palette="icefire", hue_norm=norm)
    # plt.ylim(-12, 12)

    # sns.scatterplot(x=X_valid[:, 0], y=differences[i][0], ax=ax[0, 1])
    # sns.scatterplot(x=X_valid[:, 1], y=differences[i], ax=ax[1, 0])






def predict_training(model, X_train):
    pred = model.predict(X_train)
    return pred