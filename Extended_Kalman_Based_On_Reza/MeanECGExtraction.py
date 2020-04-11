import numpy as np
from math import pi, ceil


def MeanECGExtraction(x=None, phase=None, bins=None, flag=None):
    #
    # [ECGmean,ECGsd,meanPhase] = MeanECGExtraction(x,phase,bins,flag)
    # Calculation of the mean and SD of ECG waveforms in different beats
    #
    # inputs:
    # x: input ECG signal
    # phase: ECG phase
    # bins: number of desired phase bins
    # flag
    #     1: aligns the baseline on zero, by using the mean of the first 10%
    #     segment of the calculated mean ECG beat
    #     0: no baseline alignment
    #
    # outputs:
    # ECGmean: mean ECG beat
    # ECGsd: standard deviation of ECG beats
    # meanPhase: the corresponding phase for one ECG beat

    print("-----------Inside MeanECGExtraction---------")

    meanPhase = np.zeros(bins)
    # meanPhase = meanPhase.flatten()
    ECGmean = np.zeros(bins)
    # ECGmean = ECGmean.flatten()
    ECGsd = np.zeros(bins)
    ECGsd = ECGsd.flatten()

    print(len(meanPhase), len(ECGmean), len(ECGsd))

    I = np.where((phase >= (pi - pi / bins)) | (phase < (-pi + pi / bins)))
    I = np.asarray(I).flatten()

    print("Length of Condition check", len(I))

    print("I values:", I)
    print("x values", len(x), x)
    if len(I) > 0:
        # meanPhase[0] = -pi
        # ECGmean[0] = np.mean(x[I])
        # print(np.mean(x[I]), np.std(x[I]))
        # ECGsd[0] = np.std(x[I])
        meanPhase[0] = -pi
        ECGmean[0] = np.mean(x[I])
        print(np.mean(x[I]), np.std(x[I]))
        ECGsd[0] = np.std(x[I])
    else:
        meanPhase[0] = 0
        ECGmean[0] = 0
        ECGsd[0] = -1

    for i in range(0, bins-1):
        I = np.where(
            (phase >= 2 * pi * (i+1 - 0.5) / bins - pi)
            & (phase < 2 * pi * (i+1 + 0.5) / bins - pi)
        )
        I = np.asarray(I).flatten()
        if i==1:
            print(I)
            print("++++++++++++++++++++++++++++++++++++++++++++++mean",np.mean(phase[I]), np.mean(x[I]), np.std(x[I]))
        # print("I length:Point for iteration :",i, len(I))
        if len(I) > 0:
            meanPhase[i + 1] = np.mean(phase[I])
            ECGmean[i + 1] = np.mean(x[I])
            ECGsd[i + 1] = np.std(x[I])
        else:
            meanPhase[i + 1] = 0
            ECGmean[i + 1] = 0
            ECGsd[i + 1] = -1

    print("mIDDLE:", ECGmean)

    # K = np.where(ECGsd == -1)
    # K = np.asarray(K).flatten()
    # print("K value", len(K))
    # for i in range(0, len(K)):
    #     print("here")
    #     switch = K[i]
    #     # switch0__ = K(i)
    #     if switch == 0:
    #         pass
    #     elif switch == 1:
    #         meanPhase[K[i]] = -pi
    #         ECGmean[K[i]] = ECGmean[K[i] + 1]
    #         ECGsd[K[i]] = ECGsd[K[i] + 1]
    #     elif switch == bins:
    #         meanPhase[K[i]] = pi
    #         ECGmean[K[i]] = ECGmean[K[i] - 1]
    #         ECGsd[K[i]] = ECGsd[K[i] - 1]
    #     else:
    #         meanPhase[K[i]] = np.mean(meanPhase[K[i] - 1], meanPhase[K[i] + 1])
    #         ECGmean[K[i]] = np.mean(ECGmean[K[i] - 1], ECGmean[K[i] + 1])
    #         ECGsd[K[i]] = np.mean(ECGsd[K[i] - 1], ECGsd[K[i] + 1])

    if flag == 1:
        print("Texr", np.mean(ECGmean[1 : ceil(len(ECGmean) / 10)]))
        ECGmean = ECGmean - np.mean(ECGmean[1 : ceil(len(ECGmean) / 10)])

    return ECGmean, ECGsd, meanPhase
