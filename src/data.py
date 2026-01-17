from sklearn.datasets import load_iris
import pandas as pd

def load_dataset(cfg):
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y
