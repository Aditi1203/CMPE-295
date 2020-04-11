import numpy as np
from math import radians, pi
from static_vars import static_vars
from inspect import signature

alphai = 0
bi = 0
tetai = 0
w = 0
fs = 0
dt = 0

def EKSmoother(Y=None, X0=None, P0=None, Q=None, R0=None, Wmean=None, Vmean=None, Inits=None, VarWinlen1=None, tau=None, gamma=None, VarWinlen2=None, *varargin):
    #
    # The Extended Kalman Filter (EKF) and Extended Kalman Smoother (EKS) for
    # noisy ECG observations.
    #
    # [Xekf,Pekf,Xeks,Peks,a] = EKSmoother(Y,X0,P0,Q,R,Wmean,Vmean,Inits,VarWinlen1,tau,gamma,VarWinlen2,flag),
    #
    # inputs:
    # Y: matrix of observation signals (samples x 2). First column corresponds
    # to the phase observations and the second column corresponds to the noisy
    # ECG
    # X0: initial state vector
    # P0: covariance matrix of the initial state vector
    # Q: covariance matrix of the process noise vector
    # R: covariance matrix of the observation noise vector
    # Wmean: mean process noise vector
    # Vmean: mean observation noise vector
    # Inits: filter initialization parameters
    # VarWinlen1: innovations monitoring window length
    # tau: Kalman filter forgetting time. tau=[] for no forgetting factor
    # gamma: observation covariance adaptation-rate. 0<gamma<1 and gamma=1 for no adaptation
    # VarWinlen2: window length for observation covariance adaptation
    # flag (optional): 1 with waitbar / 0 without waitbar (default)
    #
    # outputs:
    # Xekf: state vectors estimated by the EKF (samples x 2). First column
    # corresponds to the phase estimates and the second column corresponds to
    # the denoised ECG
    # Pekf: the EKF state vector covariance matrix (samples x 2 x 2)
    # Xeks: state vectors estimated by the EKS (samples x 2). First column
    # corresponds to the phase estimates and the second column corresponds to
    # the denoised ECG
    # Peks: the EKS state vector covariance matrix (samples x 2 x 2)
    # a: measure of innovations signal whiteness

    print("----------Inside EKGSmoother----------")

    plotflag = 0
    # if (nargin == 13):
    #     plotflag = varargin(1)
    plotflag=1

    # if (plotflag == 1):
    #     wtbar = waitbar(0, mstring('Forward filtering in progress. Please wait...'))

    # Initialization
    StateProp(Inits)# Initialize state equation
    global dt, alphai, bi, fs
    print("dt, fs", dt)
    ObservationProp(Inits)# Initialize output equation
    Linearization(Inits)# Initialize linearization

    # Samples = len(Y)
    # L = len(X0)
    Samples= np.size(Y, 1)
    L= np.size(X0, 0)
    print("Samples Length, XO length:", Samples, L,)
    Pminus = P0
    Xminus = X0
    Xbar = np.zeros((L, Samples))
    Pbar = np.zeros((L, L, Samples))
    Xhat = np.zeros((L, Samples))
    Phat = np.zeros((L, L, Samples))
    print("Xbar, Pbar, Xhat, Phat shape:", Xbar.shape, Pbar.shape, Xhat.shape, Phat.shape)

    # For innovation monitoring
    len_y_alongx=np.size(Y,0)
    print("VarWinlen1, VarWinlen2", VarWinlen1, VarWinlen2)
    # mem2 = np.zeros(len_y_alongx, VarWinlen2) + R0[1, 1]
    mem1 = np.ones((len_y_alongx, VarWinlen1))
    print("mem2 and mem1 type numpy",len(mem1), type(mem1))
    a = np.zeros((L, Samples))

    # Forgetting factor
    last_index=1,
    print("Inits", Inits.shape, Inits[0,1])
    fs = Inits[0,1]# the last init is fs
    dt = 1 / fs
    if (len(tau)>0):
        alpha = np.exp(-dt / tau)
    else:
        alpha = 1
    print("Forgetting Factor Values: fs, dt, alpha", fs,dt,alpha)

    #//////////////////////////////////////////////////////////////////////////
    R = R0
    # Filtering
    for k in range(0, Samples):
        # This is to prevent 'Xminus' mis-calculations on phase jumps

        if (abs(Xminus[0] - Y[0, k]) > np.round(pi,4)):
            Xminus[0]= Y[0, k]

        if(k==14999):
            print("Xminus value:", Xminus)
            print(Y[0,k])
            print(Xminus, Xminus.shape, type(Xminus))
            print(Xbar[:, k])



        # # Store results
        Xbar[:, k] = Xminus.transpose()
        # print(Xbar)
        Pbar[:, :, k] = Pminus.transpose()
        # print(Pbar)
        #
        XX = Xminus
        PP = Pminus
        Samples=2
        for jj in range(0, Samples):
            # Measurement update (A posteriori updates)
            Yminus = ObservationProp(XX, Vmean)
            YY = Yminus[jj]
            [CC, GG] = Linearization(XX, Wmean, 1)        # Linearized observation eq.
        #     C = CC(jj, [:])
        #     G = GG(jj, [:])    #----------Target   ----
#
#             K = PP * C.cT / (C * PP * C.cT + alpha * G * R(jj, jj) * G.cT)        # Kalman gain
#             PP = ((eye(L) - K * C) * PP * (eye(L) - K * C).cT + K * G * R(jj, jj) * G.cT * K.cT) / alpha        # Stabilized Kalman cov. matrix
#             XX = XX + K * (Y(jj, k) - YY)        # A posteriori state estimate
#
#         # Monitoring the innovation variance
#         inovk = Y([:], k) - Yminus
#         Yk = C * Pminus * C.cT + G * R * G.cT
#         mem1 = mcat([inovk **elpow** 2 / Yk, mem1([:], [1:end - 1])])
#         mem2 = mcat([inovk **elpow** 2, mem2([:], [1:end - 1])])
#
#         a([:], k) = mean(mem1, 2)
#
#         R(2, 2) = gamma * R(2, 2) + (1 - gamma) * mean(mem2([:], 2))
#
#         Xplus = XX
#         Pplus = (PP + PP.cT) / 2
#
#         Xminus = StateProp(Xplus, Wmean)    # State update
#         [A, F] = Linearization(Xplus, Wmean, 0)    # Linearized equations
#         Pminus = A * Pplus * A.cT + F * Q * F.cT    # Cov. matrix update
#
#         # Store results
#         Xhat([:], k) = Xplus.cT
#         Phat([:], [:], k) = Pplus.cT
#
#         if (plotflag == 1 and mod(k, Samples / 5) == 0):
#             waitbar(k / Samples, wtbar)
#
#
#     #//////////////////////////////////////////////////////////////////////////
#     if (plotflag == 1):
#         waitbar(0, wtbar, mstring('Backward smoothing in progress. Please wait...'))
#
#     # Smoothing
#     PSmoothed = zeros(size(Phat))
#     X = zeros(size(Xhat))
#     PSmoothed([:], [:], Samples) = Phat([:], [:], Samples)
#     X([:], Samples) = Xhat([:], Samples)
#     for k in [Samples - 1:-1:1]:
#         [A] = Linearization(Xhat([:], k), Wmean, 0)
#         S = Phat([:], [:], k) * A.cT / Pbar([:], [:], k + 1)
#         X([:], k) = Xhat([:], k) + S * (X([:], k + 1) - Xbar([:], k + 1))
#         PSmoothed([:], [:], k) = Phat([:], [:], k) - S * (Pbar([:], [:], k + 1) - PSmoothed([:], [:], k + 1)) * S.cT
#
#         if (plotflag == 1 and mod(k, Samples / 5) == 0):
#             waitbar(1 - k / Samples, wtbar)
#
#
#     if (plotflag == 1):
#         close(wtbar)
    return 1,2,3,4,5
#


def StateProp(x=None, Wmean=None):
    print("Inside StateProp Function")
    print("Init Value as matrix:", x.shape, x, len(x))
    x=np.asarray(x).flatten()
    print("Init Value as array:", x.shape, x, len(x))

    sig = signature(StateProp)
    params = len(sig.parameters)
    print("Total Parameters to StateProp Function:", params)

    # Check if variables should be initialized
    global alphai, bi,tetai,fs,dt
    if Wmean is None:
        # mean of the noise parameters
        print("1  named argument")
        L = int((len(x) - 2) / 3)
        print("Length L", L)
        alphai = x[0:L]
        bi = x[L + 1:2 * L]
        print("alphai, bi values: ", alphai, bi)
        tetai = x[2 * L + 1:3 * L]
        w = x[3 * L]
        fs = x[3 * L +1]
        dt = 1 / fs
        print("tetai, w, fs values:", tetai, w, fs, dt)
        return

    # print(x[0] + w * dt)
    # xout[1, 1] = x[1] + w * dt# teta state variable
#     if (xout(1, 1) > pi):
#         xout(1, 1) = xout(1, 1) - 2 * pi
#
#     dtetai = rem(xout(1, 1) - tetai, 2 * pi)
#     xout(2, 1) = x(2) - dt * sum(w * alphai /eldiv/ (bi **elpow** 2) *elmul* dtetai *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))# z state variable


def ObservationProp(x=None, v=None):

     # Check if variables should be initialized
     # print("---inside observation props---")
     # # print(v)
     # print(type(v))
     if v is None:
         return

     # Calculate output estimate
     y = np.zeros((2, 1))
     y[0] = x[0] + v[0]# teta observation
     y[1] = x[1] + v[1]# amplidute observation
     return y

#
def Linearization(x=None, Wmean=None, flag=None):

    # Make variables static
    global alphai, bi, tetai, w, fs, dt
    x = np.asarray(x).flatten()
    M=[]
    N=[]
    M = np.array(M)
    # Check if variables should be initialized
    if Wmean is None or flag is None:
        # Inits = [alphai bi tetai w fs];
        L = int((len(x) - 2) / 3)
        alphai = x[1:L]
        bi = x[L + 1:2 * L]
        tetai = x[2 * L + 1:3 * L]
        w = x[3 * L + 0]
        fs = x[3 * L + 1]
        dt = 1 / fs
        return
    # Linearize state equation
    if flag == 0:
        dtetai = (x[0] - tetai)% (2 * pi)
#
#         M[1, 1] = 1    # dF1/dteta
#         M[1, 2] = 0    # dF1/dz
# #s
#         M(2, 1) = -dt * sum(w * alphai /eldiv/ (bi **elpow** 2) *elmul* (1 - dtetai **elpow** 2. / bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))    # dF2/dteta
#         M(2, 2) = 1    # dF2/dz
#
#         # W = [alpha1, ..., alpha5, b1, ..., b5, teta1, ..., teta5, omega, N]
#         N(1, [1:3 * L]) = 0
#         N(1, 3 * L + 1) = dt
#         N(1, 3 * L + 2) = 0
#
#         N(2, [1:L]) = -dt * w /eldiv/ (bi **elpow** 2) *elmul* dtetai *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2))
#         N(2, [L + 1:2 * L]) = 2 * dt *elmul* alphai *elmul* w *elmul* dtetai /eldiv/ bi **elpow** 3. * (1 - dtetai **elpow** 2. / (2 * bi **elpow** 2)) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2))
#         N(2, [2 * L + 1:3 * L]) = dt * w * alphai /eldiv/ (bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)) *elmul* (1 - dtetai **elpow** 2. / bi **elpow** 2)
#         N(2, 3 * L + 1) = -sum(dt * alphai *elmul* dtetai /eldiv/ (bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))
#         N(2, 3 * L + 2) = 1
#
#         # Linearize output equation
    elif flag == 1:
        M[1, 1] = 1
        # M(1, 2) = 0
        # M(2, 1) = 0
        # M(2, 2) = 1
        #
        # N(1, 1) = 1
        # N(1, 2) = 0
        # N(2, 1) = 0
        # N(2, 2) = 1
   # return 1,2,3,4,5
