from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def clipper(X):
    return np.clip(X, -2, 2)

def build_numeric_preprocess():
    return Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
