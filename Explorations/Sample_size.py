from Explorations.Abstract_Harness import AbstractHarness
from Data.Generation.Abstract_DataGen import AbstractDataGen
from Data.Generation.Randomness_Gen import RandomGen
from Models.MLP_Refactor import MLP
from Models.LinReg_Basic import LinearRegression
import numpy as np

# This will explore how inherent randomness in the data
# coincides with the number of datapoints present.

class SampleSizeHarness(AbstractHarness):
    def __init__(self):
        super().__init__(dataGen=RandomGen())
        self.Model = LinearRegression()
        self.data = None
        self.do_predictions()



        # print(self.predict(np.array([[0.2, 0.2], [0.5, 0.5]])))



    def do_predictions(self):
        predictions = []
        num_points = [1, 0.5, 0.1, 0.01, 0.001]
        randomness = [0.01, 0.05, 0.1, 0.5]
        n = 10000
        self.dataGen.n_points = n
        self.dataGen.generate()
        # valid_prop = valid_samples / n
        for r in randomness:
            self.dataGen.randomness = r

            X_train, X_valid, X_test, t_train, t_valid, t_test = self.generate_data()
            for n in num_points:
                pass

        # X_train, X_valid, X_test, t_train, t_valid, t_test = self.generate_data()
        # self.train(X_train, t_train)



    def train(self, X_train, t_train):
        self.Model.train(X_train, t_train)

    def predict(self, sample):
        return self.Model.predict(sample)









if __name__ == '__main__':
    bh = SampleSizeHarness()


