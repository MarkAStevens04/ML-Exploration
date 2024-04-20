class AbstractModel:
    def __init__(self):
        self.param_combos = []
        self.Model = None
        self.training_accuracy = []
        self.validation_accuracy = []

    def fit(self):
        raise NotImplementedError

    def partial_fit(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError



