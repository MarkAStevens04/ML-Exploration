class AbstractHarness:
    def __init__(self, dataGen=None, dataFrame=None):
        self.Classes = []
        self.Model = None
        self.dataGen = dataGen
        self.dataFrame = dataFrame

    def generate_data(self):
        self.dataFrame = self.dataGen.generate()
        self.dataGen.data = self.dataFrame
        return self.dataGen.split()


    def train(self, X_train, t_train):
        raise NotImplementedError

    def vary_params(self):
        raise NotImplementedError