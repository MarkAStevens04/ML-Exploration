class AbstractHarness:
    def __init__(self):
        self.Classes = []

    def vary_params(self):
        raise NotImplementedError