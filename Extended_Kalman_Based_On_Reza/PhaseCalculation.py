import numpy as np
from math import pi


def PhaseCalculation(peaks):
    #
    # [phase phasepos] = PhaseCalculation(peaks)
    # ECG phase calculation from a given set of R-peaks.
    #
    # input:
    # peaks: vector of R-peak pulse train
    #
    # outputs:
    # phase: the calculated phases ranging from -pi to pi. The R-peaks are
    # located at phase = 0.
    # phasepos: the calculated phases ranging from 0 to 2*pi. The R-peaks are
    # again located at phasepos = 0.
    #

    print("------------Inside Phase Calculation-----------")

    N = len(peaks)
    print("Peak Length:", N)
    phasepos = np.zeros(shape=(1, N))
    phasepos = phasepos.flatten()

    I = np.where(peaks == 1)
    I = np.asarray(I).flatten()
    length_peaks = len(I)

    for i in range(0, length_peaks - 1):
        m = I[i + 1] - I[i]
        # print("Value for i, I[i], I[i+1], m:", i, I[i], I[i + 1], m)
        start = 2 * pi / m
        stop = 2 * pi
        step = 2 * pi / m
        phase = np.arange(start, stop, step)

        start_index = I[i] + 1
        end_index = I[i + 1]
        phasepos[start_index:end_index] = phase

        print(
            "Length of Arrays phase position, phase, peak phase:",
            len(phasepos[start_index:end_index]),
            len(phase),
            phasepos[end_index],
        )

    # Elements upto 1st peak
    m = I[1] - I[0]
    L = len(phasepos[0 : I[0]])
    print("Length of elements upto first phase", L)
    start = 2 * pi - (L - 1) * 2 * pi / m
    stop = 2 * pi
    step = 2 * pi / m
    print("Length of phase array", len(np.arange(start, stop, step)))
    phase = np.arange(start, stop, step)
    start_index = 1
    end_index = I[0]
    phasepos[start_index:end_index] = phase

    # Elements after last peak
    end = length_peaks - 1
    m = I[end] - I[end - 1]
    L = len(phasepos[I[end] + 1 : N - 1])
    print("Length of elements of last phase array:", L, end, m)
    start = 2 * pi / m
    stop = L * 2 * pi / m
    step = 2 * pi / m
    # print("Length of phase array", len(np.arange(start, stop + 0.0001, step)))
    phase = np.arange(start, stop + 0.0001, step)
    phasepos[I[end] + 1 : N - 1] = phase

    phasepos = np.mod(phasepos, 2 * pi)
    print("------here----", phasepos)

    phase = phasepos
    I = np.where(phasepos > pi)
    I = np.asarray(I).flatten()
    print(len(I), type(I), I)
    phase[I] = phasepos[I] - 2 * pi
    print("--------Exit Phase Calculation--------")

    return phase, phasepos, I
