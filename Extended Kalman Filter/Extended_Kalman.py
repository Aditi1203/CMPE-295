# @mfunction("Xhat, Phat, X, PSmoothed, a")
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
    #
    #
    # Open Source ECG Toolbox, version 1.0, November 2006
    # Released under the GNU General Public License
    # Copyright (C) 2006  Reza Sameni
    # Sharif University of Technology, Tehran, Iran -- LIS-INPG, Grenoble, France
    # reza.sameni@gmail.com

    # This program is free software; you can redistribute it and/or modify it
    # under the terms of the GNU General Public License as published by the
    # Free Software Foundation; either version 2 of the License, or (at your
    # option) any later version.
    # This program is distributed in the hope that it will be useful, but
    # WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
    # Public License for more details. You should have received a copy of the
    # GNU General Public License along with this program; if not, write to the
    # Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
    # MA  02110-1301, USA.

    #//////////////////////////////////////////////////////////////////////////
    plotflag = 0
    if (nargin == 13):
        plotflag = varargin(1)
    end
    if (plotflag == 1):
        wtbar = waitbar(0, mstring('Forward filtering in progress. Please wait...'))
    end
    #//////////////////////////////////////////////////////////////////////////
    # Initialization
    StateProp(Inits)# Initialize state equation
    ObservationProp(Inits)# Initialize output equation
    Linearization(Inits)# Initialize linearization

    #//////////////////////////////////////////////////////////////////////////
    Samples = length(Y)
    L = length(X0)
    Pminus = P0
    Xminus = X0
    Xbar = zeros(L, Samples)
    Pbar = zeros(L, L, Samples)
    Xhat = zeros(L, Samples)
    Phat = zeros(L, L, Samples)
    #//////////////////////////////////////////////////////////////////////////
    # For innovation monitoring
    mem2 = zeros(size(Y, 1), VarWinlen2) + R0(2, 2)
    mem1 = ones(size(Y, 1), VarWinlen1)
    a = zeros(L, Samples)
    #//////////////////////////////////////////////////////////////////////////
    # Forgetting factor
    fs = Inits(end)# the last init is fs
    dt = 1 / fs
    if (not isempty(tau)):
        alpha = exp(-dt / tau)
    else:
        alpha = 1
    end

    #//////////////////////////////////////////////////////////////////////////
    R = R0
    # Filtering
    for k in mslice[1:Samples]:

        # This is to prevent 'Xminus' mis-calculations on phase jumps
        if (abs(Xminus(1) - Y(1, k)) > pi):
            Xminus(1).lvalue = Y(1, k)
        end

        # Store results
        Xbar(mslice[:], k).lvalue = Xminus.cT
        Pbar(mslice[:], mslice[:], k).lvalue = Pminus.cT

        XX = Xminus
        PP = Pminus
        for jj in mslice[1:size(Y, 1)]:
            # Measurement update (A posteriori updates)
            Yminus = ObservationProp(XX, Vmean)
            YY = Yminus(jj)
            [CC, GG] = Linearization(XX, Wmean, 1)        # Linearized observation eq.
            C = CC(jj, mslice[:])
            G = GG(jj, mslice[:])

            K = PP * C.cT / (C * PP * C.cT + alpha * G * R(jj, jj) * G.cT)        # Kalman gain
            PP = ((eye(L) - K * C) * PP * (eye(L) - K * C).cT + K * G * R(jj, jj) * G.cT * K.cT) / alpha        # Stabilized Kalman cov. matrix
            XX = XX + K * (Y(jj, k) - YY)        # A posteriori state estimate
        end
        # Monitoring the innovation variance
        inovk = Y(mslice[:], k) - Yminus
        Yk = C * Pminus * C.cT + G * R * G.cT
        mem1 = mcat([inovk **elpow** 2 / Yk, mem1(mslice[:], mslice[1:end - 1])])
        mem2 = mcat([inovk **elpow** 2, mem2(mslice[:], mslice[1:end - 1])])

        a(mslice[:], k).lvalue = mean(mem1, 2)

        R(2, 2).lvalue = gamma * R(2, 2) + (1 - gamma) * mean(mem2(mslice[:], 2))

        Xplus = XX
        Pplus = (PP + PP.cT) / 2

        Xminus = StateProp(Xplus, Wmean)    # State update
        [A, F] = Linearization(Xplus, Wmean, 0)    # Linearized equations
        Pminus = A * Pplus * A.cT + F * Q * F.cT    # Cov. matrix update

        # Store results
        Xhat(mslice[:], k).lvalue = Xplus.cT
        Phat(mslice[:], mslice[:], k).lvalue = Pplus.cT

        if (plotflag == 1 and mod(k, Samples / 5) == 0):
            waitbar(k / Samples, wtbar)
        end
    end

    #//////////////////////////////////////////////////////////////////////////
    if (plotflag == 1):
        waitbar(0, wtbar, mstring('Backward smoothing in progress. Please wait...'))
    end

    # Smoothing
    PSmoothed = zeros(size(Phat))
    X = zeros(size(Xhat))
    PSmoothed(mslice[:], mslice[:], Samples).lvalue = Phat(mslice[:], mslice[:], Samples)
    X(mslice[:], Samples).lvalue = Xhat(mslice[:], Samples)
    for k in mslice[Samples - 1:-1:1]:
        [A] = Linearization(Xhat(mslice[:], k), Wmean, 0)
        S = Phat(mslice[:], mslice[:], k) * A.cT / Pbar(mslice[:], mslice[:], k + 1)
        X(mslice[:], k).lvalue = Xhat(mslice[:], k) + S * (X(mslice[:], k + 1) - Xbar(mslice[:], k + 1))
        PSmoothed(mslice[:], mslice[:], k).lvalue = Phat(mslice[:], mslice[:], k) - S * (Pbar(mslice[:], mslice[:], k + 1) - PSmoothed(mslice[:], mslice[:], k + 1)) * S.cT

        if (plotflag == 1 and mod(k, Samples / 5) == 0):
            waitbar(1 - k / Samples, wtbar)
        end
    end

    if (plotflag == 1):
        close(wtbar)
    end

    #//////////////////////////////////////////////////////////////////////////
    # Xhat = shiftdim(Xhat,1);
    # Phat = shiftdim(Phat,2);
    # Xbar = shiftdim(Xbar,1);
    # Pbar = shiftdim(Pbar,2);
    # X = shiftdim(X,1);
    # PSmoothed = shiftdim(PSmoothed,2);

    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
# @mfunction("xout")
def StateProp(x=None, Wmean=None):

    # Make variables static
    global __persistent__
    __persistent__['StateProp'] = 'tetai, alphai, bi, fs, w, dt'

    # Check if variables should be initialized
    if nargin == 1:
        # mean of the noise parameters
        # Inits = [alphai bi tetai w fs];
        L = (length(x) - 2) / 3
        alphai = x(mslice[1:L])
        bi = x(mslice[L + 1:2 * L])
        tetai = x(mslice[2 * L + 1:3 * L])
        w = x(3 * L + 1)
        fs = x(3 * L + 2)

        dt = 1 / fs
        return
    end

    xout(1, 1).lvalue = x(1) + w * dt# teta state variable
    if (xout(1, 1) > pi):
        xout(1, 1).lvalue = xout(1, 1) - 2 * pi
    end

    dtetai = rem(xout(1, 1) - tetai, 2 * pi)
    xout(2, 1).lvalue = x(2) - dt * sum(w * alphai /eldiv/ (bi **elpow** 2) *elmul* dtetai *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))# z state variable

    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
# @mfunction("y")
def ObservationProp(x=None, v=None):

    # Check if variables should be initialized
    if nargin == 1:
        return
    end

    # Calculate output estimate
    y = zeros(2, 1)
    y(1).lvalue = x(1) + v(1)# teta observation
    y(2).lvalue = x(2) + v(2)# amplidute observation


    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
    #//////////////////////////////////////////////////////////////////////////
# @mfunction("M, N")
def Linearization(x=None, Wmean=None, flag=None):

    # Make variables static
    global __persistent__
    __persistent__['Linearization'] = 'tetai, alphai, bi, fs, w, dt, L'

    # Check if variables should be initialized
    if nargin == 1:
        # Inits = [alphai bi tetai w fs];
        L = (length(x) - 2) / 3
        alphai = x(mslice[1:L])
        bi = x(mslice[L + 1:2 * L])
        tetai = x(mslice[2 * L + 1:3 * L])
        w = x(3 * L + 1)
        fs = x(3 * L + 2)

        dt = 1 / fs
        return
    end
    # Linearize state equation
    if flag == 0:
        dtetai = rem(x(1) - tetai, 2 * pi)

        M(1, 1).lvalue = 1    # dF1/dteta
        M(1, 2).lvalue = 0    # dF1/dz

        M(2, 1).lvalue = -dt * sum(w * alphai /eldiv/ (bi **elpow** 2) *elmul* (1 - dtetai **elpow** 2. / bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))    # dF2/dteta
        M(2, 2).lvalue = 1    # dF2/dz

        # W = [alpha1, ..., alpha5, b1, ..., b5, teta1, ..., teta5, omega, N]
        N(1, mslice[1:3 * L]).lvalue = 0
        N(1, 3 * L + 1).lvalue = dt
        N(1, 3 * L + 2).lvalue = 0

        N(2, mslice[1:L]).lvalue = -dt * w /eldiv/ (bi **elpow** 2) *elmul* dtetai *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2))
        N(2, mslice[L + 1:2 * L]).lvalue = 2 * dt *elmul* alphai *elmul* w *elmul* dtetai /eldiv/ bi **elpow** 3. * (1 - dtetai **elpow** 2. / (2 * bi **elpow** 2)) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2))
        N(2, mslice[2 * L + 1:3 * L]).lvalue = dt * w * alphai /eldiv/ (bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)) *elmul* (1 - dtetai **elpow** 2. / bi **elpow** 2)
        N(2, 3 * L + 1).lvalue = -sum(dt * alphai *elmul* dtetai /eldiv/ (bi **elpow** 2) *elmul* exp(-dtetai **elpow** 2. / (2 * bi **elpow** 2)))
        N(2, 3 * L + 2).lvalue = 1

        # Linearize output equation
    elif flag == 1:
        M(1, 1).lvalue = 1
        M(1, 2).lvalue = 0
        M(2, 1).lvalue = 0
        M(2, 2).lvalue = 1

        N(1, 1).lvalue = 1
        N(1, 2).lvalue = 0
        N(2, 1).lvalue = 0
        N(2, 2).lvalue = 1
    end