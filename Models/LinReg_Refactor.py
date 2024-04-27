from Models.Abstract_Model import AbstractModel
from sklearn.linear_model import LinearRegression



class LinReg(AbstractModel):
    def __init__(self):
        super().__init__()
        self.Model = LinearRegression()



    def train(self, X_train, t_train):
        t_train = t_train.ravel()
        self.Model.fit(X_train, t_train)

    def predict(self, sample):
        return self.Model.predict(sample)
