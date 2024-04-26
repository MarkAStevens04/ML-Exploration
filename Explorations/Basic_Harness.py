from Explorations.Abstract_Harness import AbstractHarness
from Data.Generation.Abstract_DataGen import AbstractDataGen
from Data.Generation.Basic_DataGen import BasicGen
from Models.MLP_Refactor import MLP
import numpy as np


class BasicHarness(AbstractHarness):
    def __init__(self):
        super().__init__(dataGen=BasicGen())
        self.Model = MLP()
        self.data = None
        X_train, X_valid, X_test, t_train, t_valid, t_test = self.generate_data()
        self.train(X_train, t_train)
        print(self.predict(np.array([[0.2, 0.2], [0.5, 0.5]])))
        print(f'slay inside')



    def train(self, X_train, t_train):
        self.Model.train(X_train, t_train)

    def predict(self, sample):
        return self.Model.predict(sample)









if __name__ == '__main__':
    dg = BasicGen()
    df = dg.generate()

    bh = BasicHarness()
    # X_train = np.array(df[['x', 'x2']][:90])
    # X_valid = df[['x', 'x2']][90:]
    #
    # t_train = np.array(df['y'][:90])
    # t_valid = df['y'][90:]

    # bh.train(X_train=X_train, t_train=t_train)

