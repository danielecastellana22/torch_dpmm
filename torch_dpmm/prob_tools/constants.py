import numpy as np

__all__ = ['EPS', 'LOG_2', 'LOG_PI', 'LOG_2PI', 'PI']

EPS = 10**-6
LOG_2 = np.log(2)
LOG_PI = np.log(np.pi)
LOG_2PI = np.log(2) + np.log(np.pi)
PI = np.pi