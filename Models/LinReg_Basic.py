from sklearn.linear_model import LinearRegression


def make_model(X_train, t_train):
    model = LinearRegression()
    model.fit(X=X_train, y=t_train)

    return model