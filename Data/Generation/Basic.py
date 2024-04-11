# Basic data creation methods!
import pandas as pd
import numpy as np


def create_file(dir):
    x = open(dir, 'x+')

def make_random(num_d=2, n_samples=100):
    x = pd.DataFrame(np.random.randn(100, 2), columns=['x', 'y'])
    return x
