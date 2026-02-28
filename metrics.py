import numpy as np

def MSE(modified, original):
    diff = modified.astype(float) - original.astype(float)
    return np.mean(diff**2)

