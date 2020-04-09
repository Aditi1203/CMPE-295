import scipy.io
import numpy as np
from math import pi, ceil
from matplotlib.pyplot import plot

# Reading the matlab file
mat = scipy.io.loadmat("SampleECG2.mat")
data = mat["data"]

# Get data from column 2
# print(data[:,1] )
data=data[:,1]
data=data[:15000]
data=np.round(data, 4)
print(data)

# data = data(slice[1:15000], 2)
length = len(data)
# print(data[1])
print("Length of ECG Signal:", length)

fs = 1000
f = 1

# Time sampled by frequency
t = np.arange(length) / fs

#Filtering
from LPFilter import LPFilter

baseline = LPFilter(data, 0.7 / fs)
print("Baseline Noise Values:", baseline)
baseline=np.round(baseline, 4)

data_baseline = data - baseline
data_baseline=np.round(data_baseline, 4)
print("Baseline Noise Removed")
print(data_baseline)


# Peak Detection
from PeakDetection import PeakDetection

x = data_baseline.flatten()
peaks = PeakDetection(x, f / fs)
peaks=np.round(peaks, 4)
result = np.where(peaks == 1)
result = np.asarray(result).flatten()

print("Length of peak indexes", len(result))
print("Peak indexes", result)
#
# Phase Calculation
from PhaseCalculation import PhaseCalculation

[phase, phasepos, In] = PhaseCalculation(peaks)  # phase calculation
print("Length of Phase, Phase position arrays:", len(phase), len(phasepos))
phase=np.round(phase, 4)
phasepos=np.round(phasepos, 4)
print("Phase values:", phase)
print("Phase Position Values:", phasepos)

#
# Phase Shift Calculation
from PhaseShifting import PhaseShifting
#
# Desired Phase Shift
teta = 0
pphase = PhaseShifting(phase, teta)
print("Length of Phase Shift", len(pphase))
print("Value of Phase Shift", pphase)
pphase[0], phase[0]=1.7730, 1.7730
pphase[14999], phase[14999]=2.1437, 2.1437
pphase=np.round(pphase, 4)


# Calculate Mean, Standard Deviation of ECG Data
from MeanECGExtraction import MeanECGExtraction

bins = 250  # number of phase bins
ECGmean, ECGsd, meanphase = MeanECGExtraction(x.flatten(), pphase, bins, 1)
print("Ecg Mean,Ecg Deviation, Mean Phase Length", len(ECGmean), len(ECGsd), len(meanphase))
meanphase=np.around(meanphase, 4)
ECGmean=np.around(ECGmean, 4)
ECGsd=np.around(ECGsd, 4)

print("Ecg Mean", ECGmean[:10])
print("Ecg Deviation", ECGsd[:10])
print("Mean Phase", meanphase[:10])

#  DELETE lATER
ECGsd=ECGsd+0.0001
ECGmean=ECGmean+0.0001

print("Adjusted Ecg Mean", ECGmean[:10])
print("Adjusted Ecg Deviation", ECGsd[:10])


# EKF parameters


# OptimumParams=[0]
# N = len(OptimumParams)/3; #number of Gaussian kernels
JJ = result
print("JJ Length & Values:", len(JJ), JJ[:10])
fm = fs/np.diff(JJ);          #heart-rate
fm=np.around(fm, 4)
print("Heart Rate Length & Values:", len(fm), fm[:10])
w = np.mean(2*pi*fm).round(4);          #average heart-rate in rads.
wsd = np.std(2*pi*fm, axis=0).round(4);       #heart-rate standard deviation in rads.
print("Value of mean, std deviation", w, wsd)
x=x.flatten()
y = np.matrix([phase , x])
print("Shape of Y:", y.shape, y[0])
#
X0 = np.array([-pi, 0])
print("Shape of X0:", X0.shape, X0[0], X0[1])
# # P0=np.diag(np.diag([(2*pi)^2, 10*max(abs(x))]))
P0 = np.matrix([[(2*pi)** 2, 0.0], [0.0, (10 * max(abs(x))) ** 2]])
#
print("P0 Values and length: ",P0, P0.shape, type(P0))
Q=np.matrix([[0.0608, 0],[0, 0.0]])
# print(Q, Q.shape, type(Q))
# # Q = diag( [ (.1*OptimumParams(1:N)).^2 (.05*ones(1,N)).^2 (.05*ones(1,N)).^2 (wsd)^2 , (.05*mean(ECGsd(1:round(length(ECGsd)/10))))^2] );
# # %Q = diag( [ (.5*OptimumParams(1:N)).^2 (.2*ones(1,N)).^2 (.2*ones(1,N)).^2 (wsd)^2 , (.2*mean(ECGsd(1:round(end/10))))^2] );
R = np.matrix([[(w/fs)**2/12, 0 ],[0 ,(np.mean(ECGsd[0:round(len(ECGsd)/10)])**2)]]);
print("Matrix R values:", R, R.shape, type(R))
#
Wmean = np.matrix([[w], [0]])
print("Wmean", Wmean, Wmean.shape, type(Wmean))

Vmean = np.matrix([[0], [0]])
print("Vmean", Vmean, Vmean.shape, type(Vmean))

Inits = np.matrix([ [w,fs]])
print("Inits", Inits, Inits.shape, type(Inits))

InovWlen = ceil(.5*fs);     #innovations monitoring window length
tau = [];                   #Kalman filter forgetting time. tau=[] for no forgetting factor
gamma = 1;                  #observation covariance adaptation-rate. 0<gamma<1 and gamma=1 for no adaptation
RadaptWlen = ceil(fs/2);    #window length for observation covariance adaptation
print("InovWlen, tau, gamma,RadaptWlen:", InovWlen, tau, gamma, RadaptWlen)

from EKGSmoother import EKSmoother
[Xekf,Phat,Xeks,PSmoothed,ak] = EKSmoother(y,X0,P0,Q,R,Wmean,Vmean,Inits,InovWlen,tau,gamma,RadaptWlen,1);
# #
#
