from math import cos, sqrt, pi, sin
import numpy as np
from scipy import signal
from matplotlib.pyplot import plot


def LPFilter(x, fc):
    # length = len(x[0])
    length = len(x)
    k = 0.7 # cut-off value
    print("-----Inside LP Filter------")
    alpha = (
        1
        - k * cos(2 * pi * fc)
        - sqrt(2 * k * (1 - cos(2 * pi * fc)) - k ** 2 * sin(2 * pi * fc) ** 2)
    ) / (1 - k)
    print("Alpha Value:", alpha)

    # y = [0] * length
    # xup = np.zeros(shape=(1, length))
    # for i in range(length) :
    # print(xup[0,i])
    # print(x[0,i])

    a = [1, -alpha]
    b = np.array([1 - alpha])
    print(type(b), len(b))
    # print("Length of B & a:", len(a), len(b))

    y= signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3*(max(len(b),len(a))-1))
    print("-----Exit LP Filter------")
    plot(y)
    return y
