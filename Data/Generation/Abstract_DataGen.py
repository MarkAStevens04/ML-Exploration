import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AbstractDataGen:
    def __init__(self):
        self.data = None
        self.params = None

    def generate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError