
import numpy as np

def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((x - xi)**2 + (y - yi)**2)
    weights = 1 / dist**power
    return np.sum(weights * z) / np.sum(weights)
