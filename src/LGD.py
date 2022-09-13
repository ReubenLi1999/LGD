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
import time
import os


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
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


def get_soil_moisture(background, lat_span, lon_span, resolution, flag):
    if flag == "GLDAS":
        # monthly mean GLDAS dataset
        if resolution == "monthly":
            gldas_monthly_mean = nc.Dataset(f"D:/Downloads/GLDAS/monthly/GLDAS_NOAH025_M.A{background[-6:]}.021.nc4")
        elif resolution == "daily":
            gldas_monthly_mean = nc.Dataset(f"D:/Downloads/GLDAS/three_hours/GLDAS_NOAH025_3H.A{background.strftime('%Y%m%d')}.0600.021.nc4")
        else:
            gldas_monthly_mean = []

        # load coordinates
        x = np.loadtxt("../input/x_GLDAS_0.25x0.25_3.txt")
        y = np.loadtxt("../input/y_GLDAS_0.25x0.25_3.txt")
        z = np.loadtxt("../input/z_GLDAS_0.25x0.25_3.txt")

        # soil moisture in kg*m^-2
        lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
        lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

        som_1 = np.asarray(gldas_monthly_mean["SoilMoi0_10cm_inst"][0])
        som_2 = np.asarray(gldas_monthly_mean["SoilMoi10_40cm_inst"][0])
        som_3 = np.asarray(gldas_monthly_mean["SoilMoi40_100cm_inst"][0])
        som_4 = np.asarray(gldas_monthly_mean["SoilMoi100_200cm_inst"][0])
        som = som_1 + som_2 + som_3 + som_4
    else:
        main_dir = "D:/Downloads/MERRA2/"
        # get constants
        cons = nc.Dataset(f"{main_dir}MERRA2_100.const_2d_lnd_Nx.00000000.nc4")
        # monthly mean GLDAS dataset
        if resolution == "monthly":
            gldas_monthly_mean = get_monthly_mean(2021, 6, "PRMC")
        elif resolution == "daily":
            gldas_monthly_mean = np.asarray(nc.Dataset(F"d:/Downloads/MERRA2/2021-07/MERRA2_401.tavg1_2d_lnd_Nx.{background.strftime('%Y%m%d')}.nc4")["PRMC"][0, :, :])
        else:
            gldas_monthly_mean = []

        # load coordinates
        x = np.loadtxt("../output/merra2_pr_x.txt")
        y = np.loadtxt("../output/merra2_pr_y.txt")
        z = np.loadtxt("../output/merra2_pr_z.txt")

        # soil moisture in kg*m^-2
        lat_a = np.arange(-90., 90.5, 0.5)
        lon_a = np.arange(-180., 180., 0.625)

        som = gldas_monthly_mean * np.asarray(cons["dzpr"][0, :, :]) * 1000

    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')
    # area for each segment
    res = []
    for id_lat, lat in enumerate(lat_a):
        for id_lon, lon in enumerate(lon_a):
            if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                              np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                if np.logical_and(som[id_lat, id_lon] > -500.0, som[id_lat, id_lon] < 50000.0):

                    lat_small = lat - (lat_a[2] - lat_a[1]) / 2
                    lat_large = lat + (lat_a[2] - lat_a[1]) / 2
                    lon_small = lon - (lon_a[2] - lon_a[1]) / 2
                    lon_large = lon + (lon_a[2] - lon_a[1]) / 2

                    area, _ = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                          [lat_large, lat_small, lat_small, lat_large])
                    res.append([x[id_lat, id_lon], y[id_lat, id_lon], z[id_lat, id_lon], abs(area) * som[id_lat, id_lon]])

    return np.asarray(res)


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
    if back == "background202107":
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_JPL202107.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_JPL202107.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_JPL202107.txt",
                           skiprows=6)
    elif back == "background202106":
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_JPL.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
    elif back == "background202005":
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_JPL.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_JPL.txt",
                           skiprows=6)
    else:
        sst = np.loadtxt(f"../output/grace-fo_satelliteTracking_{date4lgd}_GOCO06.txt", skiprows=6)
        pod_c = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-c_integratedOrbitFit_{date4lgd}_GOCO06.txt",
                           skiprows=6)
        pod_d = np.loadtxt(f"E:\lhsPrograms\Projects_2022\LGD\output\grace-d_integratedOrbitFit_{date4lgd}_GOCO06.txt",
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


def get_soil_moisture_diff(background, this):
    # soil moisture grid
    lat = np.loadtxt("../input/2020-07-29/lat.csv", delimiter=",")
    lon = np.loadtxt("../input/2020-07-29/lon.csv", delimiter=",")
    lat, lon = np.meshgrid(lat, lon)
    # soil moisture before the flood
    # monthly mean GLDAS dataset
    gldas_monthly_mean = nc.Dataset(f"D:/Downloads/GLDAS/monthly/GLDAS_NOAH025_M.A{background[-6:]}.021.nc4")
    gldas_daily_mean = nc.Dataset(f"D:/Downloads/GLDAS/three_hours/GLDAS_NOAH025_3H.A{this.strftime('%Y%m%d')}.0600.021.nc4")

    som_1_b = np.asarray(gldas_monthly_mean["SoilMoi0_10cm_inst"][0])
    som_2_b = np.asarray(gldas_monthly_mean["SoilMoi10_40cm_inst"][0])
    som_3_b = np.asarray(gldas_monthly_mean["SoilMoi40_100cm_inst"][0])
    som_4_b = np.asarray(gldas_monthly_mean["SoilMoi100_200cm_inst"][0])

    som_1_a = np.asarray(gldas_daily_mean["SoilMoi0_10cm_inst"][0])
    som_2_a = np.asarray(gldas_daily_mean["SoilMoi10_40cm_inst"][0])
    som_3_a = np.asarray(gldas_daily_mean["SoilMoi40_100cm_inst"][0])
    som_4_a = np.asarray(gldas_daily_mean["SoilMoi100_200cm_inst"][0])
    som_b = som_1_b + som_2_b + som_3_b + som_4_b
    som_a = som_1_a + som_2_a + som_3_a + som_4_a
    # difference
    som_diff = som_a - som_b
    np.savetxt(f"../input/soilMoisture/{this.strftime('%Y%m%d')}.csv", np.flip(som_diff, axis=0))
    som_diff[np.isclose(som_diff, 0)] = np.nan

    return lat, lon, som_diff


def compute_lgd_from_gldas(layer_1, pods_c, pods_d):
    g = 6.6743e-11
    lgd = np.zeros([pods_d.__len__(), 1])
    for i, (pod_c, pod_d) in enumerate(zip(pods_c, pods_d)):
        los = (pod_c - pod_d) / np.linalg.norm(pod_c - pod_d)
        pc = np.tile(pod_c, (layer_1.__len__(), 1))
        pd = np.tile(pod_d, (layer_1.__len__(), 1))
        pc_1 = -pc + layer_1[:, 0: 3]
        pd_1 = -pd + layer_1[:, 0: 3]
        dc_1 = np.linalg.norm(pc_1, axis=1)**3
        dd_1 = np.linalg.norm(pd_1, axis=1)**3
        f1 = pc_1 / dc_1[:, None] - pd_1 / dd_1[:, None]
        lgd[i, :] = np.dot(g * f1.T @ layer_1[:, 3], los)

    return lgd


def check_track_over_area(lat, lon, lon_span, lat_span):
    flag = False
    try:
        if np.logical_and(np.mean(lon[np.where(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]))]) <= (lon_span[1] + 2),
                          np.mean(lon[np.where(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]))]) >= (lon_span[0] - 2)):
            flag = True
    except:
        pass

    return flag


def get_soil_moisture_inc(former, latter):
    # cm = plt.cm.get_cmap("jet")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter3D(former[:, 0], former[:, 1], former[:, 2], c=latter[:, 3] - former[:, 3], cmap=cm)
    # plt.show()
    return np.c_[former[:, 0], former[:, 1], former[:, 2], latter[:, 3] - former[:, 3]]


def test_compute_lgd_from_gldas():
    layer_1 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, -1e13],
                          [6030932.53377883, 2000380.54316791, 552183.960027770, 0]])
    layer_2 = np.asarray([[6030932.53377883, 2000380.54316791, 552183.960027770, 1e13],
                          [5690954.73548881, 1887614.40476405, -2167696.78782876, 0]])
    layer_3 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, 0],
                          [5690954.73548881, 1887614.40476405, -2167696.78782876, 0]])
    layer_4 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, 0],
                          [5690954.73548881, 1887614.40476405, -2167696.78782876, 0]])
    layer_1 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, -1e13]])
    layer_2 = np.asarray([[6030932.53377883, 2000380.54316791, 552183.960027770, 1e13]])
    layer_3 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, 0]])
    layer_4 = np.asarray([[5690954.73548881, 1887614.40476405, -2167696.78782876, 0]])
    pods_c = np.asarray([[51037.0457498298, -109891.680341260, 6866882.88553822],
                         [928092.968583422, 286779.601088582, 6797535.14659767]])
    pods_d = np.asarray([[-107369.501523251, -183036.678970537, 6864808.56978168],
                         [770474.880408174, 215343.989246805, 6819942.00250571]])
    print(compute_lgd_from_gldas(layer_1, pods_c, pods_d))


def timer(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - begin_time
        print(str(func.__name__) + "函数运行时间为" + str(run_time))
        return ret
    return call_func


def get_monthly_mean(year, month, flag):
    main_dir = "D:/Downloads/MERRA2/"
    # get constants
    this_month = f"{main_dir}{year}-{month:02}/"

    # find all the data in this month
    filenames = os.listdir(this_month)
    res = np.zeros([361, 576])
    for _, filename in enumerate(filenames):
        this_day = np.asarray(nc.Dataset(f"{this_month}{filename}")[flag])
        mean = np.zeros(this_day[0, :, :].shape)
        for id in np.arange(24):
            mean = mean + this_day[id, :, :] / 24.0
        res = res + mean / filenames.__len__()

    return res


@timer
def main():
    # date array
    start_date = datetime.date(2021, 7, 2)
    end_date = datetime.date(2021, 7, 26)
    dates4lgd = [start_date + datetime.timedelta(n) for n in range(int((end_date - start_date).days))]
    # background model used
    background = "background202106"
    # hydro model
    hydro_model = "MERRA2"
    # research area
    lat_span = [31, 36.5]
    lon_span = [110, 118.5]
    lat_span_gldas = [-90, 90]
    lon_span_gldas = [-180, 180]
    # lat_span = [20, 28]
    # lon_span = [90, 97]
    # lat_span_gldas = [0, 60]
    # lon_span_gldas = [80, 110]
    # the soil moisture of the monthly mean GLDAS
    som_m = get_soil_moisture(background, lat_span_gldas, lon_span_gldas, "monthly", hydro_model)
    # coefficients
    fs_lri = 0.5
    fs_kbr = 0.2
    # date loop
    for _, date4lgd in enumerate(dates4lgd):
        kbr_x, pod_c, pod_d, gnv_c, gnv_d, lri_x, grd, sst = load_data(date4lgd, background)
        tracks_lri = divide_track(grd[::2, 1])
        tracks_kbr = divide_track(grd[::5, 1])

        sst_kbr = antialias_filter(5, sst[:, 2], "kbr")
        sst_lri = antialias_filter(2, sst[:, 2], "lri")

        rr_lri = lri_x[:, 2] + lri_x[:, 6] - sst_lri
        rr_kbr = kbr_x[:, 2] + kbr_x[:, 6] + kbr_x[:1, 9] - sst_kbr
        lgd_lri = rr2lgd(rr_lri, fs_lri, "lri")
        lgd_kbr = rr2lgd(rr_kbr, fs_kbr, "kbr")

        for ind, (span_lri, span_kbr) in enumerate(zip(tracks_lri, tracks_kbr)):
            over_area = check_track_over_area(grd[::2][span_lri[0]: span_lri[1], 1], grd[::2][span_lri[0]: span_lri[1], 0],
                                              lon_span, lat_span)
            if not over_area:
                continue
            else:
                # get the soil moisture of this day
                som_d = get_soil_moisture(date4lgd, lat_span_gldas, lon_span_gldas, "daily", hydro_model)
                lgd_gldas = compute_lgd_from_gldas(get_soil_moisture_inc(som_m, som_d),
                                                   gnv_c[span_lri[0]: span_lri[1], :],
                                                   gnv_d[span_lri[0]: span_lri[1], :])

            # plot
            fig, ax = plt.subplots(2, 2, figsize=(12, 8),
                                   gridspec_kw={'width_ratios': [40, 1], "height_ratios": [1, 1]})
            ax[1, 0].plot(grd[::2][span_lri[0]: span_lri[1], 1], lgd_lri[span_lri[0]: span_lri[1]],
                          label=f"LRI", linewidth=2, color="blue")
            ax[1, 0].plot(grd[::5][span_kbr[0]: span_kbr[1], 1], lgd_kbr[span_kbr[0]: span_kbr[1]],
                          label=f"KBR", linewidth=2, color="green")
            ax[1, 0].plot(grd[::2][span_lri[0]: span_lri[1], 1], lgd_gldas,
                          label=f"GLDAS", linewidth=2, color="red")
            # # get data from first line of the plot
            # newx = ax[1].lines[1].get_ydata()
            # newy = ax[1].lines[1].get_xdata()
            # # set new x- and y- data for the line
            # ax[1].lines[1].set_xdata(newx)
            # ax[1].lines[1].set_ydata(newy)
            ax[1, 0].set_ylim([-5e-9, 5e-9])
            ax[1, 0].yaxis.get_offset_text().set_fontsize(24)
            ax[1, 0].axvspan(lat_span[0], lat_span[1], alpha=0.5, color='red')
            ax[1, 0].tick_params(labelsize=25, width=2.9)
            ax[1, 0].legend(fontsize=15, loc='best', frameon=False)
            ax[1, 0].set_ylabel('LGD [m/s$^2$]', fontsize=20)
            ax[1, 0].set_xlabel('Lat. [deg]', fontsize=20)
            ax[1, 0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
            plt.setp(ax[1, 0].spines.values(), linewidth=3)
            plt.setp(ax[0, 0].spines.values(), linewidth=3)

            ax[0, 0].set_title(f"{date4lgd}-{ind}", fontsize=24)
            cm = plt.cm.get_cmap("jet")
            # get the soil moisture
            lat, lon, som_diff = get_soil_moisture_diff(background, date4lgd)
            im = ax[0, 0].pcolormesh(lon, lat, som_diff.T, cmap=cm)
            ax[0, 0].plot(grd[::2][span_lri[0]: span_lri[1], 0], grd[::2][span_lri[0]: span_lri[1], 1], color="purple", linewidth=2)
            ax[0, 0].tick_params(labelsize=25, width=2.9)
            ax[0, 0].yaxis.get_offset_text().set_fontsize(24)
            ax[0, 0].add_patch(Rectangle((lon_span[0], lat_span[0]),
                                         lon_span[1] - lon_span[0],
                                         lat_span[1] - lat_span[0],
                                         fill=None, alpha=1))
            ax[0, 0].set_ylabel('Lat. [deg]', fontsize=20)
            ax[0, 0].set_xlabel('Lon. [deg]', fontsize=20)
            ax[0, 0].scatter(113.68, 34.75, color="k", marker="*")
            ax[0, 0].axis('equal')
            ax[0, 0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
            cbar = fig.colorbar(im, cax=ax[0, 1])
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_title('EWH [mm]', fontsize=15)
            ax[0, 0].set_ylim([20, 60])
            ax[1, 1].axis("off")
            plt.tight_layout()
            plt.savefig(f"../image/{background}/lgd_{date4lgd}_{ind}_{hydro_model}.png", dpi=600)

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
        # plt.savefig(f"../image/{background}/lgd_{date4lgd}_ASD.png", dpi=600)
        # plt.show()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    main()
    # test_compute_lgd_from_gldas()
