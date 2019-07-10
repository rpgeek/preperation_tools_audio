
import scipy.io.wavfile as wavfile
import numpy as np
import multiprocessing as mp
import random
import librosa

def abs_normalize_wave_minmax(x):
    x = x.astype(np.int32)
    imax = np.max(np.abs(x))
    x_n = x / imax
    return x_n 

def abs_short_normalize_wave_minmax(x):
    imax = 32767.
    x_n = x / imax
    return x_n 

def dynamic_normalize_wave_minmax(x):
    x = x.astype(np.int32)
    imax = np.max(x)
    imin = np.min(x)
    x_n = (x - np.min(x)) / (float(imax) - float(imin))
    return x_n * 2 - 1

def normalize_wave_minmax(x):
    return (2./65535.) * (x - 32767.) + 1.

def pre_emphasize(x, coef=0.95):
    if coef <= 0:
        return x
    x0 = np.reshape(x[0], (1,))
    diff = x[1:] - coef * x[:-1]
    concat = np.concatenate((x0, diff), axis=0)
    return concat

def de_emphasize(y, coef=0.95):
    if coef <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coef * x[n - 1] + y[n]
    return x
