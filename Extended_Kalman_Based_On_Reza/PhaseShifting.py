from math import pi
import numpy as np


def PhaseShifting(phasein=None, teta=None):
    #
    # phase = PhaseShifting(phasein,teta),
    # Phase shifter.
    #
    # inputs:
    # phasein: calculated ECG phase
    # teta: desired phase shift. teta>0 and teta<0 corresponds with phase leads
    # and phase lags, respectively.
    #
    # output:
    # phase: the shifted phase.
    #

    phase = phasein + teta
    phase = np.mod(phase + pi, 2 * pi) - pi

    return phase
