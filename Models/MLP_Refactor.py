from Models.Abstract_Model import AbstractModel
from sklearn.neural_network import MLPRegressor



class MLP(AbstractModel):
    def __init__(self, hidden_shape=(1), max_iter=1000, activation='relu'):
        super().__init__()
        self.Model = MLPRegressor(hidden_layer_sizes=hidden_shape, max_iter=max_iter, activation=activation)

    def train(self, X_train, t_train):
        t_train = t_train.ravel()
        self.Model.fit(X_train, t_train)

    def predict(self, sample):
        return self.Model.predict(sample)
