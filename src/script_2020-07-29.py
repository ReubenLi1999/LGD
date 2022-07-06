import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, windows, kaiserord, firwin
from scipy.fftpack import rfft, irfft, fftfreq
from pyproj import Geod
import astropy.coordinates as ac
from astropy import units as u
from astropy.time import Time
from scipy.sparse.linalg import lsqr


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
    lgd_filter = np.zeros(w.size)
    lgd_filter[1: ] = 0.000345 * np.power(w[1: ], -1.04) + 1
    lgd_filter[(w < 1e-3)] = 1
    filtered = f_signal * lgd_filter
    cut = irfft(filtered)
    cut = butter_highpass_filter(cut, 2e-3, fs, 5)
    return cut


def lgd2mass(source_locations, coor_c, coor_d, los, lgd, lat, flag):

    # resample
    source_lat = source_locations[:, 0].reshape(41, 6)  # resample
    source_lat = source_lat[::4, ::2]
    source_lon = source_locations[:, 1].reshape(41, 6)  # resample
    source_lon = source_lon[::4, ::2]
    source_locations = np.c_[source_lat.reshape(source_lat.shape[0]*source_lat.shape[1], 1),
                             source_lon.reshape(source_lon.shape[0]*source_lon.shape[1], 1)]
    # fig, ax = plt.subplots(figsize=(16, 8))
    # plt.scatter(source_locations[:, 1], source_locations[:, 0])
    # plt.show()

    # convert geocentric to itrs
    itrs = ac.ITRS(ac.WGS84GeodeticRepresentation(ac.Longitude(source_locations[:, 1], unit=u.degree),
                                                  ac.Latitude(source_locations[:, 0], unit=u.degree),
                                                  u.Quantity(12, unit=u.m)),
                   obstime=Time("2000-01-01T00:00:00"))
    q = ac.EarthLocation(*itrs.cartesian.xyz)
    r_1 = coor_c[:, 0: 3]
    r_2 = coor_d[:, 0: 3]
    e = los

    # arc for computation
    if flag == "a":
        sta = np.argmax(lat > 21.5)
        end = np.argmin(lat < 26)
    else:
        end = np.argmin(lat > 22.5)
        sta = np.argmax(lat < 27)

    # constant
    G = 6.6743e-11
    e = G * e

    # f loop
    mass = np.zeros([los.__len__(), q.__len__()])
    for id_r, (r_1_j, r_2_j, e_j, lgd_j) in enumerate(zip(r_1, r_2, e, lgd)):
        f_1 = np.zeros([q.__len__(), 3])
        f_2 = np.zeros([q.__len__(), 3])
        a = np.zeros([q.__len__(), 1])
        for id_q, q_i in enumerate(q):
            q_i = np.asarray([q_i.x.value, q_i.y.value, q_i.z.value])
            f_1[id_q, :] = (q_i - r_1_j) / np.linalg.norm(q_i - r_1_j)**3
            f_2[id_q, :] = (q_i - r_2_j) / np.linalg.norm(q_i - r_2_j)**3
            mass[id_r, id_q] = np.dot(e_j, f_1[id_q, :] - f_2[id_q, :])

    m = lsqr(mass[sta: end, :], lgd[sta: end])[0]

    return np.sum(m)


def soil_moisture(lat_span, lon_span, flag):
    # soil moisture in kg*m^-2
    lat = np.loadtxt("../input/2020-07-29/lat.csv", delimiter=",")
    lon = np.loadtxt("../input/2020-07-29/lon.csv", delimiter=",")
    if flag == "a":
        som_1 = np.loadtxt("../input/2020-07-29/SoilMoi0_10cm_inst.csv", delimiter=",")
        som_2 = np.loadtxt("../input/2020-07-29/SoilMoi10_40cm_inst.csv", delimiter=",")
        som_3 = np.loadtxt("../input/2020-07-29/SoilMoi40_100cm_inst.csv", delimiter=",")
        som_4 = np.loadtxt("../input/2020-07-29/SoilMoi100_200cm_inst.csv", delimiter=",")
    else:
        som_1 = np.loadtxt("../input/2020-07-29/SoilMoi0_10cm_inst_de.csv", delimiter=",")
        som_2 = np.loadtxt("../input/2020-07-29/SoilMoi10_40cm_inst_de.csv", delimiter=",")
        som_3 = np.loadtxt("../input/2020-07-29/SoilMoi40_100cm_inst_de.csv", delimiter=",")
        som_4 = np.loadtxt("../input/2020-07-29/SoilMoi100_200cm_inst_de.csv", delimiter=",")
    som = som_1 + som_2 + som_3 + som_4
    som_insitu = som[np.where(np.logical_and(lat>=lat_span[0], lat<=lat_span[1])), :][0]
    som_insitu = som_insitu[:, np.where(np.logical_and(lon>=lon_span[0], lon<=lon_span[1]))[0]]
    lat_insitu = lat[np.where(np.logical_and(lat>=lat_span[0], lat<=lat_span[1]))[0]]
    lon_insitu = lon[np.where(np.logical_and(lon>=lon_span[0], lon<=lon_span[1]))[0]]
    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')
    # area for each segment
    res = []
    for id_lat, lat in enumerate(lat_insitu):
        for id_lon, lon in enumerate(lon_insitu):
            if not np.isnan(som_insitu[id_lat, id_lon]):
                lat_small = lat - 0.125
                lat_large = lat + 0.125
                lon_small = lon - 0.125
                lon_large = lon + 0.125
                area, perim = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                          [lat_large, lat_small, lat_small, lat_large])
                res.append([lat, lon, abs(area) * som_insitu[id_lat, id_lon]])
    return np.asarray(res)


def ascending():

    # the starting epoch of the short arc on 2020-07-29 is 10629 [2s sampling]

    # Bangladesh in latitude and longitude
    bang = np.asarray([[3.59686092e-10, 5874119.6544634, 2476724.01886489]])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    lri_x = np.loadtxt("../input/2020-07-29/LRI1B_2020-07-29_Y_04.txt", dtype=np.longdouble, skiprows=0)
    pod_c = np.loadtxt("../output/ddk7_C_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    pod_d = np.loadtxt("../output/ddk7_D_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    gnv_c = np.loadtxt("../input/2020-07-29/GNV1B_2020-07-29_C_04.txt", skiprows=148, usecols=[3, 4, 5])[::2]
    gnv_d = np.loadtxt("../input/2020-07-29/GNV1B_2020-07-29_D_04.txt", skiprows=148, usecols=[3, 4, 5])[::2]
    lgd_out = np.loadtxt("../input/2020-07-29/lgd20200729", dtype=np.longdouble)
    latlon = np.loadtxt("../input/2020-07-29/coor_2020-07-29_ascend.txt", dtype=np.longdouble)

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

    fig, ax = plt.subplots(figsize=(16, 8))
    rr = lri_x[:, 2] - range_rate_pod[:, 0]
    lgd = rr2lgd(rr, fs)
    plt.plot(lgd)
    plt.plot(lgd_out[:, 2])
    ax.tick_params(labelsize=25, width=2.9)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    ax.set_title("2020-07-29", fontsize=24)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)

    fig, ax = plt.subplots(figsize=(16, 8))  # 2020-07-29
    plt.plot(latlon[:, 0], lgd[10629: 10629+1421], label="self")
    plt.plot(latlon[:, 0], lgd_out[10629: 10629+1421, 0], label="released(open-access)")
    pod_c_arc = gnv_c[10629: 10629+1421]
    pod_d_arc = gnv_d[10629: 10629+1421]
    los_arc = los[10629: 10629+1421]
    lgd_arc = lgd[10629: 10629+1421]
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('Lat [deg]', fontsize=20)
    ax.set_title("2020-07-29", fontsize=24)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)

    # fig, ax = plt.subplots(figsize=(16, 8))  # 2021-07-21
    # plt.plot(latlon[:, 0], lgd[24880: 24880+1420: 5])
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('Lat [deg]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)

    # fig, ax = plt.subplots(figsize=(16, 8))  # 2021-03-22
    # plt.plot(latlon[:, 0], lgd[36415: 36415+1415])
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('Lat [deg]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)

    # mass of sources
    solm = soil_moisture([22, 26], [88, 92], "a")
    print("Mass of soil moisture: ", np.sum(solm[:, 2]), "kg.")
    mass = lgd2mass(solm[:, :2], pod_c_arc, pod_d_arc, los_arc, lgd_arc, latlon[:, 0], "a")
    print((mass - np.sum(solm[:, 2]))/1e12)

    # ASD of LGD
    X, f, C = lpsd(lgd, windows.nuttall, 1e-4, 2e-1, 400, 100, 2, fs, 0.5)
    fig, ax = plt.subplots(figsize=(16, 8))
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

    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.loglog(freq_lri_x_range,
    #              np.sqrt(psd_lri_x_range),
    #              linewidth=4,
    #              label='lri-range',
    #              color='xkcd:aqua',)
    # ax.loglog(freq_pod_x_range,
    #              np.sqrt(psd_pod_x_range),
    #              linewidth=4,
    #              label='lri-range-residual',
    #              color='#9C27B0',)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('频率 [Hz]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'ASD [m/$\sqrt{Hz}$]', fontsize=20)
    # ax.legend(fontsize=15, loc='best', frameon=False)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)
    # plt.tight_layout()

    # fig, ax = plt.subplots(figsize=(16, 8))
    # ax.loglog(freq_lri_x_rate,
    #              np.sqrt(psd_lri_x_rate),
    #              linewidth=4,
    #              label='lri-range-rate',
    #              color='xkcd:aqua',)
    # ax.loglog(freq_pod_x_rate,
    #              np.sqrt(psd_pod_x_rate) ,
    #              linewidth=4,
    #              label='lri-range-rate-residual',
    #              color='#9C27B0',)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.set_xlabel('频率 [Hz]', fontsize=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.set_ylabel(r'ASD [m/s/$\sqrt{Hz}$]', fontsize=20)
    # ax.legend(fontsize=15, loc='best', frameon=False)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)
    # plt.tight_layout()


def descending():

    # the starting epoch of the short arc on 2020-07-29 is 31904 [2s sampling]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    lri_x = np.loadtxt("../input/2020-07-29/LRI1B_2020-07-29_Y_04.txt", dtype=np.longdouble, skiprows=0)
    pod_c = np.loadtxt("../output/ddk7_C_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    pod_d = np.loadtxt("../output/ddk7_D_2020-07-29.txt", dtype=np.longdouble, skiprows=6)
    gnv_c = np.loadtxt("../input/2020-07-29/GNV1B_2020-07-29_C_04.txt", skiprows=148, usecols=[3, 4, 5])[::2]
    gnv_d = np.loadtxt("../input/2020-07-29/GNV1B_2020-07-29_D_04.txt", skiprows=148, usecols=[3, 4, 5])[::2]
    lgd_out = np.loadtxt("../input/2020-07-29/lgd20200729", dtype=np.longdouble)
    latlon = np.loadtxt("../input/2020-07-29/coor_2020-07-29_descend.txt", dtype=np.longdouble)

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
    rr = lri_x[:, 2] - range_rate_pod[:, 0]
    lgd = rr2lgd(rr, fs)

    fig, ax = plt.subplots(figsize=(16, 8))  # 2020-07-29
    plt.plot(latlon[:, 0], lgd[31904: 31904+1419], label="self")
    plt.plot(latlon[:, 0], lgd_out[31904: 31904+1419, 0], label="released(open-access)")
    pod_c_arc = gnv_c[31904: 31904+1419]
    pod_d_arc = gnv_d[31904: 31904+1419]
    los_arc = los[31904: 31904+1419]
    lgd_arc = lgd[31904: 31904+1419]
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('Lat [deg]', fontsize=20)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel(r'LGD [m/s$^2\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    # plt.show()

    # mass of sources
    solm = soil_moisture([22, 26], [88, 92], "d")
    print("Mass of soil moisture: ", np.sum(solm[:, 2]), "kg.")
    mass = lgd2mass(solm[:, :2], pod_c_arc, pod_d_arc, los_arc, lgd_arc, latlon[:, 0], "d")
    print((mass - np.sum(solm[:, 2]))/1e12)


if __name__ == "__main__":
    ascending()
    descending()
