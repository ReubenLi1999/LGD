import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, windows, kaiserord, firwin
from scipy.fftpack import rfft, irfft, fftfreq
from pyproj import Geod
import astropy.coordinates as ac
from astropy import units as u
from astropy.time import Time
from scipy.sparse.linalg import lsqr
from matplotlib.patches import Rectangle
import datetime
import netCDF4 as nc


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


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


def rr2lgd(rr, fs, flag):

    nans, x = nan_helper(rr)
    try:
        rr[nans] = np.interp(x(nans), x(~nans), rr[~nans])
    except:
        rr[:] = np.nan

    ra = np.gradient(rr) * fs
    w = fftfreq(ra.size, d=1/fs)
    f_signal = rfft(ra)
    lgd_filter = np.zeros(w.size)
    lgd_filter[1: ] = 0.000345 * np.power(w[1: ], -1.04) + 1
    lgd_filter[(w < 1e-3)] = 1
    filtered = f_signal * lgd_filter
    cut = irfft(filtered)
    if flag == "lri":
        coeff = np.loadtxt("../input/highpass.fcf", dtype=np.float64, skiprows=14)
    else:
        coeff = np.loadtxt("../input/bandpass_1mhz_10mhz_kbr.fcf", dtype=np.float64, skiprows=14)
    cut = filtfilt(coeff, 1, cut)
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
        sta = np.argmax(lat > 22.5)
        end = np.argmin(lat < 27)
    else:
        end = np.argmin(lat > 21)
        sta = np.argmax(lat < 25)

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


def get_soil_moisture(background, lat_span, lon_span):
    # monthly mean GLDAS dataset
    gldas_monthly_mean = nc.Dataset(f"D:/Downloads/GLDAS/GLDAS_monthlymean/GLDAS_NOAH025_M.A{background[-6:]}.021.nc4")

    # coordinates of the grid points of GLDAS
    # layer 1
    x_1 = np.loadtxt("../input/x_GLDAS_0.25x0.25_1.txt")
    y_1 = np.loadtxt("../input/y_GLDAS_0.25x0.25_1.txt")
    z_1 = np.loadtxt("../input/z_GLDAS_0.25x0.25_1.txt")
    # layer 2
    x_2 = np.loadtxt("../input/x_GLDAS_0.25x0.25_2.txt")
    y_2 = np.loadtxt("../input/y_GLDAS_0.25x0.25_2.txt")
    z_2 = np.loadtxt("../input/z_GLDAS_0.25x0.25_2.txt")
    # layer 3
    x_3 = np.loadtxt("../input/x_GLDAS_0.25x0.25_3.txt")
    y_3 = np.loadtxt("../input/y_GLDAS_0.25x0.25_3.txt")
    z_3 = np.loadtxt("../input/z_GLDAS_0.25x0.25_3.txt")
    # layer 4
    x_4 = np.loadtxt("../input/x_GLDAS_0.25x0.25_4.txt")
    y_4 = np.loadtxt("../input/y_GLDAS_0.25x0.25_4.txt")
    z_4 = np.loadtxt("../input/z_GLDAS_0.25x0.25_4.txt")

    # soil moisture in kg*m^-2
    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

    som_1 = np.asarray(gldas_monthly_mean["SoilMoi0_10cm_inst"][0])
    som_2 = np.asarray(gldas_monthly_mean["SoilMoi10_40cm_inst"][0])
    som_3 = np.asarray(gldas_monthly_mean["SoilMoi40_100cm_inst"][0])
    som_4 = np.asarray(gldas_monthly_mean["SoilMoi100_200cm_inst"][0])
    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')
    # area for each segment
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    for id_lat, lat in enumerate(lat_a):
        for id_lon, lon in enumerate(lon_a):
            lat_small = lat - 0.125
            lat_large = lat + 0.125
            lon_small = lon - 0.125
            lon_large = lon + 0.125
            if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                              np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                if som_1[id_lat, id_lon] > -100.0:
                    area, _ = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                              [lat_large, lat_small, lat_small, lat_large])
                    res1.append([x_1[id_lon, id_lat], y_1[id_lon, id_lat], z_1[id_lon, id_lat], abs(area) * som_1[id_lat, id_lon]])
                    res2.append([x_2[id_lon, id_lat], y_2[id_lon, id_lat], z_2[id_lon, id_lat], abs(area) * som_2[id_lat, id_lon]])
                    res3.append([x_3[id_lon, id_lat], y_3[id_lon, id_lat], z_3[id_lon, id_lat], abs(area) * som_3[id_lat, id_lon]])
                    res4.append([x_4[id_lon, id_lat], y_4[id_lon, id_lat], z_4[id_lon, id_lat], abs(area) * som_4[id_lat, id_lon]])

    return np.asarray(res1), np.asarray(res2), np.asarray(res3), np.asarray(res4)


def antialias_filter(interval, sig, flag):
    if flag == 'lri':
        coeff = np.loadtxt("../input/dealias_lri.fcf", skiprows=14)
    else:
        coeff = np.loadtxt("../input/dealias_kbr.fcf", skiprows=14)

    cut = filtfilt(coeff, 1, sig)

    return cut[::interval]


def divide_track(lat):
    ind = np.zeros([lat.__len__()+1, 1])
    ind[np.where(np.diff(lat) > 0)[0]] = 1
    ind[-2] = ind[-3]
    ind[-1] = ind[-3]
    ind_p = np.where(np.diff(ind[:, 0]) > 0.5)[0]
    ind_n = np.where(np.diff(ind[:, 0]) < -0.5)[0]
    temp = np.append(ind_p, ind_n)
    temp = np.append(temp, [0])
    ind = np.sort(temp)
    ind = np.append(ind, [-1])

    res = np.zeros([ind.__len__()-1, 2], dtype=int)
    for id in np.arange(0, ind.__len__()-1, 1):
        res[id, 0] = ind[id]
        res[id, 1] = ind[id+1]

    return res


def adjust_sst(data, flag):
    # create the time series
    if flag == "lri":
        time = np.arange(0, 86400, 2) + data[0, 0]
    else:
        time = np.arange(0, 86400, 5) + data[0, 0]

    res = np.zeros([time.__len__(), data.shape[1]])
    res[:] = np.nan
    for _, dat in enumerate(data):
        res[np.where(time == dat[0]), :] = dat

    return res


def load_data(date4lgd, back):

    kbr_x = np.loadtxt(f"E:\lhsPrograms\gracefo_dataset\gracefo_1B_{date4lgd}_RL04.ascii.noLRI\KBR1B_{date4lgd}_Y_04.txt",
                       dtype=np.float64,
                       skiprows=162)
    if back == "background202106":
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_JPL.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
    else:
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_JPL202107.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_JPL202107.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_JPL202107.txt",
                           skiprows=6)
    gnv_c = np.loadtxt(f"E:\lhsPrograms\gracefo_dataset\gracefo_1B_{date4lgd}_RL04.ascii.noLRI\GNV1B_{date4lgd}_C_04.txt",
                       skiprows=148,
                       usecols=[3, 4, 5])[::2]
    gnv_d = np.loadtxt(f"E:\lhsPrograms\gracefo_dataset\gracefo_1B_{date4lgd}_RL04.ascii.noLRI\GNV1B_{date4lgd}_D_04.txt",
                       skiprows=148,
                       usecols=[3, 4, 5])[::2]
    try:
        lri_x = np.loadtxt(f"E:\lhsPrograms\gracefo_dataset\gracefo_1B_{date4lgd}_RL04.ascii.LRI\LRI1B_{date4lgd}_Y_04.txt",
                           skiprows=129,)
        lri_x = adjust_sst(lri_x, "lri")
    except:
        lri_x = np.empty([43200, 7])
        lri_x[:] = np.nan

    grd_c = np.loadtxt(f"../input/demo/grace-c_approximateOrbit_{date4lgd}.txt", skiprows=4)
    grd_d = np.loadtxt(f"../input/demo/grace-d_approximateOrbit_{date4lgd}.txt", skiprows=4)
    grd = (grd_d + grd_c) / 2.

    kbr_x = adjust_sst(kbr_x, "kbr")

    return kbr_x, pod_c, pod_d, gnv_c, gnv_d, lri_x, grd, sst


def get_soil_moisture_diff():
    # soil moisture grid
    lat = np.loadtxt("../input/2020-07-29/lat.csv", delimiter=",")
    lon = np.loadtxt("../input/2020-07-29/lon.csv", delimiter=",")
    Lat, Lon = np.meshgrid(lat, lon)
    # soil moisture before the flood
    som_1_b = np.loadtxt("../input/2021-07-01/SoilMoi0_10cm_inst.csv", delimiter=",")
    som_2_b = np.loadtxt("../input/2021-07-01/SoilMoi10_40cm_inst.csv", delimiter=",")
    som_3_b = np.loadtxt("../input/2021-07-01/SoilMoi40_100cm_inst.csv", delimiter=",")
    som_4_b = np.loadtxt("../input/2021-07-01/SoilMoi100_200cm_inst.csv", delimiter=",")
    som_b = som_1_b + som_2_b + som_3_b + som_4_b
    # soil moisture after the flood
    som_1_a = np.loadtxt("../input/2021-07-28/SoilMoi0_10cm_inst.csv", delimiter=",")
    som_2_a = np.loadtxt("../input/2021-07-28/SoilMoi10_40cm_inst.csv", delimiter=",")
    som_3_a = np.loadtxt("../input/2021-07-28/SoilMoi40_100cm_inst.csv", delimiter=",")
    som_4_a = np.loadtxt("../input/2021-07-28/SoilMoi100_200cm_inst.csv", delimiter=",")
    som_a = som_1_a + som_2_a + som_3_a + som_4_a
    # difference
    som_diff = som_a - som_b

    return Lat, Lon, som_diff


def compute_lgd_from_gldas(layer_1, layer_2, layer_3, layer_4, pods_c, pods_d, los):
    g = 6.6743 * 10e-11
    lgd = np.zeros([pods_d.__len__(), 1])
    for i, (pod_c, pod_d) in enumerate(zip(pods_c, pods_d)):
        pc = np.tile(pod_c[1: 4], (layer_1.__len__(), 1))
        pd = np.tile(pod_d[1: 4], (layer_1.__len__(), 1))
        pc_1 = pc - layer_1[:, 0: 3]
        pc_2 = pc - layer_2[:, 0: 3]
        pc_3 = pc - layer_3[:, 0: 3]
        pc_4 = pc - layer_4[:, 0: 3]
        pd_1 = pd - layer_1[:, 0: 3]
        pd_2 = pd - layer_2[:, 0: 3]
        pd_3 = pd - layer_3[:, 0: 3]
        pd_4 = pd - layer_4[:, 0: 3]
        dc_1 = np.linalg.norm(pc_1, axis=1)**3
        dc_2 = np.linalg.norm(pc_2, axis=1)**3
        dc_3 = np.linalg.norm(pc_3, axis=1)**3
        dc_4 = np.linalg.norm(pc_4, axis=1)**3
        dd_1 = np.linalg.norm(pd_1, axis=1)**3
        dd_2 = np.linalg.norm(pd_2, axis=1)**3
        dd_3 = np.linalg.norm(pd_3, axis=1)**3
        dd_4 = np.linalg.norm(pd_4, axis=1)**3
        f1 = -pc_1 / dc_1[:, None] + pd_1 / dd_1[:, None]
        f2 = -pc_2 / dc_2[:, None] + pd_2 / dd_2[:, None]
        f3 = -pc_3 / dc_3[:, None] + pd_3 / dd_3[:, None]
        f4 = -pc_4 / dc_4[:, None] + pd_4 / dd_4[:, None]
        lgd[i] = np.dot(g * f1.T @ layer_1[:, 3], los[i, :]) \
               + np.dot(g * f2.T @ layer_2[:, 3], los[i, :]) \
               + np.dot(g * f3.T @ layer_3[:, 3], los[i, :]) \
               + np.dot(g * f4.T @ layer_4[:, 3], los[i, :])

    return lgd

    # for j, (l1, l2, l3, l4) in enumerate(zip(layer_1, layer_2, layer_3, layer_4)):
    #     pc_1 = pod_c[1: 4] - l1[0: 3]
    #     pc_2 = pod_c[1: 4] - l2[0: 3]
    #     pc_3 = pod_c[1: 4] - l3[0: 3]
    #     pc_4 = pod_c[1: 4] - l4[0: 3]
    #     pd_1 = pod_d[1: 4] - l1[0: 3]
    #     pd_2 = pod_d[1: 4] - l2[0: 3]
    #     pd_3 = pod_d[1: 4] - l3[0: 3]
    #     pd_4 = pod_d[1: 4] - l4[0: 3]
    #     dc_1 = np.linalg.norm(pc_1)
    #     dc_2 = np.linalg.norm(pc_2)
    #     dc_3 = np.linalg.norm(pc_3)
    #     dc_4 = np.linalg.norm(pc_4)
    #     dd_1 = np.linalg.norm(pd_1)
    #     dd_2 = np.linalg.norm(pd_2)
    #     dd_3 = np.linalg.norm(pd_3)
    #     dd_4 = np.linalg.norm(pd_4)
    #     f1[:, j] = pc_1 / dc_1**3 - pd_1 / dd_1**3
    #     f2[:, j] = pc_2 / dc_2**3 - pd_2 / dd_2**3
    #     f3[:, j] = pc_3 / dc_3**3 - pd_3 / dd_3**3
    #     f4[:, j] = pc_4 / dc_4**3 - pd_4 / dd_4**3


def check_track_over_area(lat, lon, lon_span, lat_span):
    flag = False
    try:
        if np.logical_and(np.mean(lon[np.where(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]))]) <= (lon_span[1] + 2),
                          np.mean(lon[np.where(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]))]) >= (lon_span[0] - 2)):
            flag = True
    except:
        pass

    return flag


def main():
    # date array
    start_date = datetime.date(2021, 7, 22)
    end_date = datetime.date(2021, 7, 23)
    dates4lgd = [start_date + datetime.timedelta(n) for n in range(int((end_date - start_date).days))]
    # background model used
    background = "background202106"
    # research area
    lat_span = [31, 36.5]
    lon_span = [110, 118.5]
    lat_span_gldas = [-60, 90]
    lon_span_gldas = [-180, 180]
    # get the soil moisture
    lat, lon, som_diff = get_soil_moisture_diff()
    # the soil moisture of the monthly mean GLDAS
    layer_1, layer_2, layer_3, layer_4 = get_soil_moisture(background, lat_span_gldas, lon_span_gldas)
    # coefficients
    fs_lri = 0.5
    fs_kbr = 0.2
    # date loop
    for _, date4lgd in enumerate(dates4lgd):
        kbr_x, pod_c, pod_d, gnv_c, gnv_d, lri_x, grd, sst = load_data(date4lgd, background)
        tracks_lri = divide_track(grd[::2, 1])
        tracks_kbr = divide_track(grd[::5, 1])
        tracks_gld = divide_track(grd[::10, 1])

        pos_rela = pod_c[:, 1: 4] - pod_d[:, 1: 4]
        dis = np.zeros([pos_rela.__len__(), 1])
        los = np.zeros([pos_rela.__len__(), 3])
        for index, e in enumerate(pos_rela):
            dis[index] = np.linalg.norm(e)
            los[index, :] = pos_rela[index, :] / dis[index]

        sst_kbr = antialias_filter(5, sst[:, 2], "kbr")
        sst_lri = antialias_filter(2, sst[:, 2], "lri")

        rr_lri = lri_x[:, 2] + lri_x[:, 6] - sst_lri
        rr_kbr = kbr_x[:, 2] + kbr_x[:, 6] + kbr_x[:1, 9] - sst_kbr
        lgd_lri = rr2lgd(rr_lri, fs_lri, "lri")
        lgd_kbr = rr2lgd(rr_kbr, fs_kbr, "kbr")

        for ind, (span_lri, span_kbr, span_gld) in enumerate(zip(tracks_lri, tracks_kbr, tracks_gld)):
            over_area = check_track_over_area(grd[::2][span_lri[0]: span_lri[1], 1], grd[::2][span_lri[0]: span_lri[1], 0],
                                              lon_span, lat_span)
            if not over_area:
                continue
            # plot
            fig, ax = plt.subplots(2, 1, figsize=(16, 8))
            ax[1].plot(grd[::2][span_lri[0]: span_lri[1], 1], lgd_lri[span_lri[0]: span_lri[1]], label=f"LRI-{ind}",
                       linewidth=2, color="blue")
            ax[1].plot(grd[::5][span_kbr[0]: span_kbr[1], 1], lgd_kbr[span_kbr[0]: span_kbr[1]], label=f"KBR-{ind}",
                       linewidth=2, color="green")
            if over_area:
                lgd_gldas = compute_lgd_from_gldas(layer_1, layer_2, layer_3, layer_4,
                                                   pod_c[::10][span_gld[0]: span_gld[1], :],
                                                   pod_d[::10][span_gld[0]: span_gld[1], :],
                                                   los[::10][span_gld[0]: span_gld[1], :])
                ax[1].plot(grd[::10][span_gld[0]: span_gld[1], 1], lgd_gldas, label=f"GLDAS-{ind}",
                           linewidth=2, color="red")
            ax[1].tick_params(labelsize=25, width=2.9)
            ax[1].yaxis.get_offset_text().set_fontsize(24)
            ax[1].legend(fontsize=15, loc='best', frameon=False)
            ax[1].set_ylabel(r'LGD [m/s$^2$]', fontsize=20)
            ax[1].set_xlabel('Lat [°]', fontsize=20)
            ax[1].set_ylim([-3e-9, 3e-9])
            ax[1].axvspan(lat_span[0], lat_span[1], alpha=0.5, color='red')
            ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
            plt.setp(ax[1].spines.values(), linewidth=3)
            plt.setp(ax[0].spines.values(), linewidth=3)

            ax[0].set_title(f"{date4lgd}-{ind}", fontsize=24)
            cm = plt.cm.get_cmap("jet")
            ax[0].pcolormesh(lon, lat, som_diff.T, cmap=cm)
            ax[0].plot(grd[::2][span_lri[0]: span_lri[1], 0], grd[::2][span_lri[0]: span_lri[1], 1], color="purple", linewidth=2)
            ax[0].tick_params(labelsize=25, width=2.9)
            ax[0].yaxis.get_offset_text().set_fontsize(24)
            ax[0].add_patch(Rectangle((lon_span[0], lat_span[0]),
                                      lon_span[1] - lon_span[0],
                                      lat_span[1] - lat_span[0],
                                      fill=None, alpha=1))
            ax[0].set_ylabel(r'Latitude [$\degree$]', fontsize=20)
            ax[0].set_xlabel(r'Longitude [$\degree$]', fontsize=20)
            ax[0].scatter(113.68, 34.75, color="k", marker="*")
            ax[0].axis('equal')
            ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
            ax[0].set_ylim([30, 40])
            if over_area:
                ax[0].add_patch(Rectangle((lon_span_gldas[0], lat_span_gldas[0]),
                                          lon_span_gldas[1] - lon_span_gldas[0],
                                          lat_span_gldas[1] - lat_span_gldas[0],
                                          fill=None, alpha=1, linestyle="dashed"))
                plt.show()
                exit()

            plt.tight_layout()
            plt.savefig(f"../image/{background}/lgd_{date4lgd}_{ind}.png", dpi=600)

        # ASD of LGD
        x_lri, f_lri, c_lri = lpsd(lgd_lri, windows.nuttall, 1e-4, 2e-1, 400, 100, 2, fs_lri, 0.5)
        x_kbr, f_kbr, c_kbr = lpsd(lgd_kbr, windows.nuttall, 1e-4, 1e-1, 400, 100, 2, fs_kbr, 0.5)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.loglog(f_lri,
                  np.sqrt(x_lri*c_lri['PSD']) * 1e9,
                  linewidth=4,
                  color='blue',
                  label="LLGD")
        ax.loglog(f_kbr,
                  np.sqrt(x_kbr*c_kbr['PSD']) * 1e9,
                  linewidth=4,
                  color='green',
                  label="KLGD")
        ax.tick_params(labelsize=25, width=2.9)
        ax.set_xlabel('Frequency [Hz]', fontsize=20)
        ax.set_title(f"{date4lgd}", fontsize=20)
        ax.set_xlim([1e-4, 5e-1])
        ax.yaxis.get_offset_text().set_fontsize(24)
        ax.set_ylabel(r'LGD [nm/s$^2 / \sqrt{Hz}$]', fontsize=20)
        ax.legend(fontsize=15, loc='best', frameon=False)
        ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
        plt.setp(ax.spines.values(), linewidth=3)
        plt.tight_layout()
        plt.savefig(f"../image/{background}/lgd_{date4lgd}_ASD.png", dpi=600)


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    main()
