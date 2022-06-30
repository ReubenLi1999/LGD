import numpy as np
import matplotlib.pyplot as plt
from lpsd import lpsd_trad
from scipy.signal import welch, butter, filtfilt, windows, kaiserord, firwin
from scipy.fftpack import rfft, irfft, fftfreq


def lpsd(x, windowfcn, fmin, fmax, Jdes, Kdes, Kmin, fs, xi):
    """
    LPSD Power spectrum estimation with a logarithmic frequency axis.
    Estimates the power spectrum or power spectral density of the time series x at JDES frequencies equally spaced (on
    a logarithmic scale) from FMIN to FMAX.
    Originally at: https://github.com/tobin/lpsd
    Translated from Matlab to Python by Rudolf W Byker in 2018.
    The implementation follows references [1] and [2] quite closely; in
    particular, the variable names used in the program generally correspond
    to the variables in the paper; and the corresponding equation numbers
    are indicated in the comments.
    References:
        [1] Michael Tröbs and Gerhard Heinzel, "Improved spectrum estimation
        from digitized time series on a logarithmic frequency axis," in
        Measurement, vol 39 (2006), pp 120-129.
            * http://dx.doi.org/10.1016/j.measurement.2005.10.010
            * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1
        [2] Michael Tröbs and Gerhard Heinzel, Corrigendum to "Improved
        spectrum estimation from digitized time series on a logarithmic
        frequency axis."
    Author(s): Tobin Fricke <tobin.fricke@ligo.org> 2012-04-17
    :param x: time series to be transformed. "We assume to have a long stream x(n), n=0, ..., N-1 of equally spaced
        input data sampled with frequency fs. Typical values for N range from 10^4 to >10^6" - Section 8 of [1]
    :param windowfcn: function handle to windowing function (e.g. @hanning) "Choose a window function w(j, l) to reduce
        spectral leakage within the estimate. ... The computations of the window function will be performed when the
        segment lengths L(j) have been determined." - Section 8 of [1]
    :param fmin: lowest frequency to estimate. "... we propose not to use the first few frequency bins. The first
        frequency bin that yields unbiased spectral estimates depends on the window function used. The bin is given by
        the effective half-width of the window transfer function." - Section 7 of [1].
    :param fmax: highest frequency to estimate
    :param Jdes: desired number of Fourier frequencies. "A typical value for J is 1000" - Section 8 of [1]
    :param Kdes: desired number of averages
    :param Kmin: minimum number of averages
    :param fs: sampling rate
    :param xi: fractional overlap between segments (0 <= xi < 1). See Figures 5 and 6. "The amount of overlap is a
        trade-off between computational effort and flatness of the data weighting." [1]
    :return: Pxx, f, C
        - Pxx: vector of (uncalibrated) power spectrum estimates
        - f: vector of frequencies corresponding to Pxx
        - C: dict containing calibration factors to calibrate Pxx into either power spectral density or power spectrum.
    """

    # Sanity check the input arguments
    if not callable(windowfcn):
        raise TypeError("windowfcn must be callable")
    if not (fmax > fmin):
        raise ValueError("fmax must be greater than fmin")
    if not (Jdes > 0):
        raise ValueError("Jdes must be greater than 0")
    if not (Kdes > 0):
        raise ValueError("Kdes must be greater than 0")
    if not (Kmin > 0):
        raise ValueError("Kmin must be greater than 0")
    if not (Kdes >= Kmin):
        raise ValueError("Kdes must be greater than or equal to Kmin")
    if not (fs > 0):
        raise ValueError("fs must be greater than 0")
    if not (0 <= xi < 1):
        raise ValueError("xi must be: 0 <= xi 1")

    N = len(x)  # Table 1
    jj = np.arange(Jdes, dtype=int)  # Table 1

    if not (fmin >= float(fs) / N):  # Lowest frequency possible
        raise ValueError("The lowest possible frequency is {}, but fmin={}".format(float(fs) / N), fmin)
    if not (fmax <= float(fs) / 2):  # Nyquist rate
        raise ValueError("The Nyquist rate is {}, byt fmax={}".format(float(fs) / 2, fmax))

    g = np.log(fmax) - np.log(fmin)  # (12)
    f = fmin * np.exp(jj * g / float(Jdes - 1))  # (13)
    rp = fmin * np.exp(jj * g / float(Jdes - 1)) * (np.exp(g / float(Jdes - 1)) - 1)  # (15)

    # r' now contains the 'desired resolutions' for each frequency bin, given the rule that we want the resolution to be
    # equal to the difference in frequency between adjacent bins. Below we adjust this to account for the minimum and
    # desired number of averages.

    ravg = (float(fs) / N) * (1 + (1 - xi) * (Kdes - 1))  # (16)
    rmin = (float(fs) / N) * (1 + (1 - xi) * (Kmin - 1))  # (17)

    case1 = rp >= ravg  # (18)
    case2 = np.logical_and(rp < ravg, np.sqrt(ravg * rp) > rmin)  # (18)
    case3 = np.logical_not(np.logical_or(case1, case2))  # (18)

    rpp = np.zeros(Jdes)

    rpp[case1] = rp[case1]  # (18)
    rpp[case2] = np.sqrt(ravg * rp[case2])  # (18)
    rpp[case3] = rmin  # (18)

    # r'' contains adjusted frequency resolutions, accounting for the finite length of the data, the constraint of the
    # minimum number of averages, and the desired number of averages.  We now round r'' to the nearest bin of the DFT
    # to get our final resolutions r.
    L = np.around(float(fs) / rpp).astype(int)  # segment lengths (19)
    r = float(fs) / L  # actual resolution (20)
    m = f / r  # Fourier Tranform bin number (7)

    Pxx = np.zeros(Jdes)
    S1 = np.zeros(Jdes)
    S2 = np.zeros(Jdes)

    # Loop over frequencies.  For each frequency, we basically conduct Welch's method with the fourier transform length
    # chosen differently for each frequency.
    # TODO: Try to eliminate the for loop completely, since it is unpythonic and slow. Maybe write doctests first...
    for jj in range(len(f)):
        # Calculate the number of segments
        D = int(np.around((1 - xi) * L[jj]))  # (2)
        K = int(np.floor((N - L[jj]) / float(D) + 1))  # (3)

        # reshape the time series so each column is one segment  <-- FIXME: This is not clear.
        a = np.arange(L[jj])
        b = D * np.arange(K)
        ii = a[:, np.newaxis] + b  # Selection matrix
        data = x[ii]  # x(l+kD(j)) in (5)

        # Remove the mean of each segment.
        data = data - np.mean(data, axis=0)  # (4) & (5)
        if jj == 0:
            print(np.mean(data))

        # Compute the discrete Fourier transform
        window = windowfcn(L[jj]+2)[1:-1]  # (5) #signal.hann is equivalent to Matlab hanning, however, the first and the last elements are zeros, need to be removed
        window = window[:, np.newaxis]

        sinusoid = np.exp(-2j * np.pi * np.arange(L[jj])[:, np.newaxis] * m[jj] / L[jj])  # (6)
        data = data * (sinusoid * window)  # (5,6)

        # Average the squared magnitudes
        Pxx[jj] = np.mean(np.abs(np.sum(data, axis=0)) ** 2)  # (8) #python sum over column should be np.sum(data, axis=0) insteads of np.sum(data)

        # Calculate some properties of the window function which will be used during calibration
        S1[jj] = sum(window)  # (23)
        S2[jj] = sum(window ** 2)  # (24)

    # Calculate the calibration factors
    C = {
        'PS': 2. / (S1 ** 2),  # (28)
        'PSD': 2. / (fs * S2)  # (29)
    }

    return Pxx, f, C


def kaiser(x, fq, cutoff_hz, ripple_db=600.):
    # the desired width of the transition from pass to stop
    width = 0.12 / fq

    # the desired attenuation in the stop band in db: ripple_db

    # compute the kaiser parameter for the fir filter
    n, beta = kaiserord(ripple_db, width)
    print('The length of the lowpass filter is ', n, '.')

    # use firwin with a kaiser window
    taps = firwin(n,
                  cutoff_hz,
                  window=('kaiser', beta),
                  pass_zero='lowpass',
                  nyq=fq)

    # use filtfilt to filter x with the fir filter
    filtered_x = filtfilt(taps, 1.0, x)

    return filtered_x


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def rr2lgd(rr, fs):
    ra = np.gradient(rr) * fs
    w = fftfreq(ra.size, d=1/fs)
    f_signal = rfft(ra)
    lgd_filter = 0.000345 * w**(-1.04) + 1
    lgd_filter[(w < 1e-3)] = 1
    filtered = f_signal * lgd_filter
    cut = irfft(filtered)
    cut = butter_highpass_filter(cut, 2e-3, fs, 5)
    return cut


def main():

    # the starting epoch of the short arc on 2020-07-29 is 10631 [2s sampling]

    # Bangladesh in latitude and longitude
    bang = np.asarray([7.08123588e-10, 11564535.81122418, 5374575.99816614])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    lri_x = np.loadtxt("../input/2020-07-29/LRI1B_2020-07-29_Y_04.txt", dtype=np.longdouble, skiprows=0)
    pod_c = np.loadtxt("../output/temp_C_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    pod_d = np.loadtxt("../output/temp_D_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    lgd_out = np.loadtxt("../input/2020-07-29/lgd20200729", dtype=np.longdouble)
    latlon = np.loadtxt("../input/2020-07-29/coor_2020-07-29.txt", dtype=np.longdouble)

    vec_rela = pod_c[:, 4: 7] - pod_d[:, 4: 7]
    pos_rela = pod_c[:, 1: 4] - pod_d[:, 1: 4]
    dis = np.zeros([pos_rela.__len__(), 1])
    los = np.zeros([pos_rela.__len__(), 3])
    range_rate_pod = np.zeros([pos_rela.__len__(), 1])
    for index, e in enumerate(pos_rela):
        dis[index] = np.sqrt(pos_rela[index, 0]**2 + pos_rela[index, 1]**2 + pos_rela[index, 2]**2)
        los[index, :] = pos_rela[index, :] / dis[index]
        range_rate_pod[index] = np.dot(vec_rela[index, :], los[index, :])

    fs = 0.5

    freq_lri_x_range, psd_lri_x_range = welch(lri_x[:, 1], fs, ('kaiser', 30.), lri_x.__len__(), scaling='density')
    freq_lri_x_rate, psd_lri_x_rate = welch(lri_x[:, 2], fs, ('kaiser', 30.), lri_x.__len__(), scaling='density')
    freq_pod_x_range, psd_pod_x_range = welch(lri_x[:, 1] - dis[:, 0], fs, ('kaiser', 30.), lri_x.__len__(), scaling='density')
    freq_pod_x_rate, psd_pod_x_rate = welch(lri_x[:, 2] - range_rate_pod[:, 0], fs, ('kaiser', 30.), lri_x.__len__(), scaling='density')

    fig, ax = plt.subplots(figsize=(10, 5))
    rr = lri_x[:, 2] - range_rate_pod[:, 0]
    lgd = rr2lgd(rr, fs)
    plt.plot(lgd)
    plt.plot(lgd_out[:, 2])
    ax.tick_params(labelsize=25, width=2.9)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)

    fig, ax = plt.subplots(figsize=(10, 5))  # 2020-07-29
    plt.plot(latlon[:, 0], lgd[10631: 10631+1420: 5], label="self")
    plt.plot(latlon[:, 0], lgd_out[10631: 10631+1420: 5, 0], label="released(open-access)")
    pod_c_arc = pod_c[10631: 10631+1420]
    pod_d_arc = pod_d[10631: 10631+1420]
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('纬度 [deg]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)

    # fig, ax = plt.subplots(figsize=(10, 5))  # 2021-07-21
    # plt.plot(latlon[:, 0], lgd[24880: 24880+1420: 5])
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('纬度 [deg]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)

    # fig, ax = plt.subplots(figsize=(10, 5))  # 2021-03-22
    # plt.plot(latlon[:, 0], lgd[36415: 36415+1415])
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('纬度 [deg]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freq_lri_x_range,
                 np.sqrt(psd_lri_x_range),
                 linewidth=4,
                 label='lri-range',
                 color='xkcd:aqua',)
    ax.loglog(freq_pod_x_range,
                 np.sqrt(psd_pod_x_range),
                 linewidth=4,
                 label='lri-range-residual',
                 color='#9C27B0',)
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('频率 [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_ylabel(r'ASD [m/$\sqrt{Hz}$]', fontsize=20)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freq_lri_x_rate,
                 np.sqrt(psd_lri_x_rate),
                 linewidth=4,
                 label='lri-range-rate',
                 color='xkcd:aqua',)
    ax.loglog(freq_pod_x_rate,
                 np.sqrt(psd_pod_x_rate) ,
                 linewidth=4,
                 label='lri-range-rate-residual',
                 color='#9C27B0',)
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('频率 [Hz]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_ylabel(r'ASD [m/s/$\sqrt{Hz}$]', fontsize=20)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()

    X, f, C = lpsd(lgd, windows.nuttall, 1e-4, 2e-1, 400, 100, 2, fs, 0.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(f,
                np.sqrt(X*C['PSD']) * 1e9,
                linewidth=4,
                color='#9C27B0',
                label="LRI")
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('频率 [Hz]', fontsize=20)
    ax.set_xlim([1e-4, 5e-1])
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_ylabel(r'LGD [nm/s$^2 / \sqrt{Hz}$]', fontsize=20)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
