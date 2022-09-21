import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from pyproj import Geod
from matplotlib import path
import pandas as pd
import os
import datetime


def get_gldas_series_monthly():
    # study area
    lat_span = [30, 40]
    lon_span = [110, 120]
    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')
    # soil moisture in kg*m^-2
    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

    filenames_monthly_mean = ["202101", "202102", "202103", "202104", "202105", "202106",
                              "202107", "202108", "202109", "202110", "202111", "202112"]
    gldas_monthly_means = np.zeros([12, 2])
    for id_monthly, (_, filename) in enumerate(zip(gldas_monthly_means, filenames_monthly_mean)):
        gldas_monthly_means[id_monthly, 0] = id_monthly + 1
        # count points
        count = 0

        gldas = nc.Dataset(f"D:/Downloads/GLDAS/monthly/GLDAS_NOAH025_M.A{filename}.021.nc4")
        som = np.asarray(gldas["SoilMoi0_10cm_inst"][0]) + np.asarray(gldas["SoilMoi10_40cm_inst"][0]) \
            + np.asarray(gldas["SoilMoi40_100cm_inst"][0]) + np.asarray(gldas["SoilMoi100_200cm_inst"][0])
        for id_lat, lat in enumerate(lat_a):
            for id_lon, lon in enumerate(lon_a):
                if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                                  np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                    if som[id_lat, id_lon] > -500.0:
                        count = count + 1
                        lat_small = lat - 0.125
                        lat_large = lat + 0.125
                        lon_small = lon - 0.125
                        lon_large = lon + 0.125

                        area, _ = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                              [lat_large, lat_small, lat_small, lat_large])
                        gldas_monthly_means[id_monthly, 1] = gldas_monthly_means[id_monthly, 1] + abs(area) * som[id_lat, id_lon]

        gldas_monthly_means[id_monthly, 1] = gldas_monthly_means[id_monthly, 1] / count

    return gldas_monthly_means


def get_gldas_series_daily():
    # study area
    lat_span = [30, 40]
    lon_span = [110, 120]
    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')
    # soil moisture in kg*m^-2
    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

    filenames_monthly_mean = os.listdir("D:/Downloads/GLDAS/three_hours/")

    gldas_monthly_means = np.zeros([filenames_monthly_mean.__len__(), 2])
    for id_monthly, (_, filename) in enumerate(zip(gldas_monthly_means, filenames_monthly_mean)):
        gldas_monthly_means[id_monthly, 0] = float(filename[-17: -15]) + float(filename[-15: -13]) / 30 \
                                             + float(filename[-12: -10]) / 30 / 24
        # count points
        count = 0

        gldas = nc.Dataset(f"D:/Downloads/GLDAS/three_hours/{filename}")
        som = np.asarray(gldas["SoilMoi0_10cm_inst"][0]) + np.asarray(gldas["SoilMoi10_40cm_inst"][0]) \
            + np.asarray(gldas["SoilMoi40_100cm_inst"][0]) + np.asarray(gldas["SoilMoi100_200cm_inst"][0])
        for id_lat, lat in enumerate(lat_a):
            for id_lon, lon in enumerate(lon_a):
                if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                                  np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                    if som[id_lat, id_lon] > -500.0:
                        count = count + 1
                        lat_small = lat - 0.125
                        lat_large = lat + 0.125
                        lon_small = lon - 0.125
                        lon_large = lon + 0.125

                        area, _ = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                              [lat_large, lat_small, lat_small, lat_large])
                        gldas_monthly_means[id_monthly, 1] = gldas_monthly_means[id_monthly, 1] + abs(area) * som[id_lat, id_lon]
        gldas_monthly_means[id_monthly, 1] = gldas_monthly_means[id_monthly, 1] / count

    return gldas_monthly_means


def check_water_balance():
    # study area
    # lat_span = [30, 40]
    lat_span = [33, 35]
    # lon_span = [110, 120]
    lon_span = [112, 115]
    # soil moisture in kg*m^-2
    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

    filenames_monthly_mean = os.listdir("D:/Downloads/GLDAS/three_hours/")

    som_mean = np.zeros([480, 2])
    som_baln = np.zeros([480, 2])
    precipitation = np.zeros([480, 2])
    evaporation = np.zeros([480, 2])
    surface_runoff = np.zeros([480, 2])
    subterranean_runoff = np.zeros([480, 2])
    for id_monthly, (_, filename) in enumerate(zip(som_mean, filenames_monthly_mean)):
        som_mean[id_monthly, 0] = float(filename[-17: -15]) + float(filename[-15: -13]) / 31 \
                                             + float(filename[-12: -10]) / 31 / 24
        som_baln[id_monthly, 0] = float(filename[-17: -15]) + float(filename[-15: -13]) / 31 \
                                             + float(filename[-12: -10]) / 31 / 24
        # count points
        count = 0

        gldas = nc.Dataset(f"D:/Downloads/GLDAS/three_hours/{filename}")
        som = np.asarray(gldas["SoilMoi0_10cm_inst"][0]) + np.asarray(gldas["SoilMoi10_40cm_inst"][0]) \
            + np.asarray(gldas["SoilMoi40_100cm_inst"][0]) + np.asarray(gldas["SoilMoi100_200cm_inst"][0])
        eva = np.asarray(gldas["Evap_tavg"][0]) * 3 * 60 * 60
        per = np.asarray(gldas["Rainf_f_tavg"][0]) * 3 * 60 * 60
        sur = np.asarray(gldas["Qs_acc"][0])
        sub = np.asarray(gldas["Qsb_acc"][0])
        for id_lat, lat in enumerate(lat_a):
            for id_lon, lon in enumerate(lon_a):
                if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                                  np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                    if som[id_lat, id_lon] > -500.0:
                        count = count + 1
                        # som_mean[id_monthly, 1] = som_mean[id_monthly, 1] + som[id_lat, id_lon]
                        # som_baln[id_monthly, 1] = som_baln[id_monthly, 1] + per[id_lat, id_lon] - eva[id_lat, id_lon] \
                        #                         - sur[id_lat, id_lon] - sub[id_lat, id_lon]
                        precipitation[id_monthly, 1] = precipitation[id_monthly, 1] + per[id_lat, id_lon]
                        evaporation[id_monthly, 1] = evaporation[id_monthly, 1] + eva[id_lat, id_lon]
                        surface_runoff[id_monthly, 1] = surface_runoff[id_monthly, 1] + sur[id_lat, id_lon]
                        subterranean_runoff[id_monthly, 1] = subterranean_runoff[id_monthly, 1] + sub[id_lat, id_lon]

        # som_mean[id_monthly, 1] = som_mean[id_monthly, 1] / count
        # som_baln[id_monthly, 1] = som_baln[id_monthly, 1] / count
        surface_runoff[id_monthly, 1] = surface_runoff[id_monthly, 1] / count
        precipitation[id_monthly, 1] = precipitation[id_monthly, 1] / count
        evaporation[id_monthly, 1] = evaporation[id_monthly, 1] / count
        subterranean_runoff[id_monthly, 1] = subterranean_runoff[id_monthly, 1] / count

    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.step(som_mean[0: -1, 0], np.cumsum(np.diff(som_mean[:, 1])), label="sum_of_soil_moisture")
    ax.plot(som_baln[:, 0], precipitation[:, 1], label="precipitation")
    ax.plot(som_baln[:, 0], evaporation[:, 1], label="evaporation")
    ax.plot(som_baln[:, 0], surface_runoff[:, 1], label="surface_runoff")
    ax.plot(som_baln[:, 0], subterranean_runoff[:, 1], label="subterranean_runoff")
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.tick_params(labelsize=25, width=2.9)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel('kg/m$^2$', fontsize=20)
    ax.set_xlabel('Month of 2021', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    np.savetxt("../tmp/precipitation.txt", precipitation)
    np.savetxt("../tmp/evaporation.txt", evaporation)
    np.savetxt("../tmp/surface_runoff.txt", surface_runoff)
    np.savetxt("../tmp/subterranean_runoff.txt", subterranean_runoff)

    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.step(som_mean[0: -1, 0], np.cumsum(np.diff(som_mean[:, 1])), label="sum_of_soil_moisture")
    ax.step(som_baln[:, 0], np.cumsum(precipitation[:, 1]), label="precipitation")
    ax.step(som_baln[:, 0], np.cumsum(evaporation[:, 1]), label="evaporation")
    ax.step(som_baln[:, 0], np.cumsum(surface_runoff[:, 1]), label="surface_runoff")
    ax.step(som_baln[:, 0], np.cumsum(subterranean_runoff[:, 1]), label="subterranean_runoff")
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.tick_params(labelsize=25, width=2.9)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel('kg/m$^2$', fontsize=20)
    ax.set_xlabel('Month of 2021', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


def absoluteFilePaths(directory):
    res = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            res.append(os.path.abspath(os.path.join(dirpath, f)))

    return res


def check_discharge():
    coors_stations = np.asarray([[112.24, 34.5], [116.18, 34.5]])
    y = np.loadtxt("../input/yellow_river_subregion.txt")
    p = path.Path(y)
    num = 8 * 61

    areas = np.loadtxt("../input/areas_GLDAS.txt")
    areas[np.isnan(areas)] = 0
    discharges_measured = np.zeros([61, coors_stations.shape[0]])
    filenames_measured = os.listdir("D:/Downloads/dailyReportYellowRiver")
    for id, _ in enumerate(discharges_measured):
        tmp = np.asarray(pd.read_excel(f"D:/Downloads/dailyReportYellowRiver/{filenames_measured[id]}"))
        # discharges_measured[id, 0] = tmp[43, 4]
        discharges_measured[id, 0] = tmp[29, 4]
        discharges_measured[id, 1] = tmp[50, 4]

    # soil moisture in kg*m^-2
    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

    filenames = absoluteFilePaths("D:/Downloads/GLDAS/three_hours")[0: num]

    runoff = np.zeros([num, coors_stations.shape[0]])
    precip = np.zeros([num, coors_stations.shape[0]])
    is_in_polygon = np.zeros([coors_stations.shape[0], lat_a.__len__(), lon_a.__len__()])

    for id_lat, lat in enumerate(lat_a):
        for id_lon, lon in enumerate(lon_a):
            for id_sta, coor_station in enumerate(coors_stations):
                if np.logical_and(p.contains_points([[lon, lat]])[0], lon < coor_station[0] - (lon_a[2] - lon_a[1]) / 2):
                    is_in_polygon[id_sta, id_lat, id_lon] = 1

    for id_sta, coor_station in enumerate(coors_stations):
        for id_monthly, filename in enumerate(filenames):
            # if id_monthly > 408:
            #     continue
            gldas = nc.Dataset(filename)
            sur = np.asarray(gldas["Qs_acc"][0])
            sub = np.asarray(gldas["Qsb_acc"][0])
            pre = np.asarray(gldas["Rainf_f_tavg"][0])
            sur[sur == -9999] = 0
            sub[sub == -9999] = 0
            pre[pre == -9999] = 0
            sur = (sur + sub) * is_in_polygon[id_sta, :, :] * areas / 1000 / (3 * 60 * 60)
            pre = pre * is_in_polygon[id_sta, :, :] * areas / 1000
            runoff[id_monthly, id_sta] = np.sum(sur)
            precip[id_monthly, id_sta] = np.sum(pre)

    fig, ax = plt.subplots(figsize=(12, 8))
    cm = plt.cm.get_cmap("jet")
    lat, lon = np.meshgrid(lat_a, lon_a)
    im = ax.pcolormesh(lon, lat, np.log10(pre.T), cmap=cm)
    ax.plot(y[:, 0], y[:, 1])
    ax.axis("equal")
    cbar = fig.colorbar(im)
    cbar.ax.set_title('', fontsize=15)
    plt.show()

    runoff = np.add.reduceat(runoff, np.arange(0, len(runoff), 8)) / 8
    precip = np.add.reduceat(precip, np.arange(0, len(precip), 8)) / 8
    np.savetxt("../output/GLDAS_discharge.txt", runoff)
    np.savetxt("../output/GLDAS_precipitation.txt", precip)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.linspace(6, 8, runoff.__len__()), runoff[:, 0], label="GLDAS_runoff", linewidth=2, marker="o")
    ax.plot(np.linspace(6, 8, runoff.__len__()), precip[:, 0], label="GLDAS_precipitation", linewidth=2, marker="o")
    ax.plot(np.linspace(6, 8, discharges_measured.__len__()), discharges_measured[:, 0], label="measured_discharge", linewidth=2, marker="o")
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.tick_params(labelsize=25, width=2.9)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel('Discharge [m$^3$/s]', fontsize=20)
    ax.set_xlabel('Month of 2021', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # monthly = get_gldas_series_monthly()
    # daily = get_gldas_series_daily()

    # fig, ax = plt.subplots(figsize=(12, 8))
    # ax.scatter(daily[:, 0], daily[:, 1], label="3_hours_mean")
    # ax.scatter(monthly[:, 0], monthly[:, 1], label="monthly_mean", marker="*", s=20)
    # ax.yaxis.get_offset_text().set_fontsize(24)
    # ax.tick_params(labelsize=25, width=2.9)
    # ax.legend(fontsize=15, loc='best', frameon=False)
    # ax.set_ylabel('Mass [kg]', fontsize=20)
    # ax.set_xlabel('Month of 2021', fontsize=20)
    # ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    # plt.setp(ax.spines.values(), linewidth=3)
    # plt.tight_layout()
    # plt.show()
    # check_water_balance()
    check_discharge()
