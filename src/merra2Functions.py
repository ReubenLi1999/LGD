import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import geojson
from matplotlib.patches import Polygon
from matplotlib import path
import pandas as pd
from pyproj import Geod


def main():
    demo = nc.Dataset("D:/Downloads/MERRA2_401.tavg1_2d_lnd_Nx.20210702.nc4")
    print(np.asarray(demo["PRMC"][0, :, :]).shape)


def get_monthly_mean(year, month, flag):
    main_dir = "D:/Downloads/MERRA2/"
    this_month = f"{main_dir}{year}-{month:02}/"

    # find all the data in this month
    filenames = os.listdir(this_month)
    res = np.zeros([361, 576])
    for _, filename in enumerate(filenames):
        this_day = np.asarray(nc.Dataset(f"{this_month}{filename}")[flag])
        mean = np.zeros(this_day[0, :, :].shape)
        for id in np.arange(12):
            mean = mean + this_day[id, :, :] / 12.0
        res = res + mean / filenames.__len__()

    return res


def absoluteFilePaths(directory):
    res = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            res.append(os.path.abspath(os.path.join(dirpath, f)))

    return res


def check_water_balance():
    # study area
    lat_span = [33, 35]
    lon_span = [112, 115]
    # soil moisture in kg*m^-2
    lat_a = np.arange(-90., 90.5, 0.5)
    lon_a = np.arange(-180., 180., 0.625)

    main_dir = "D:/Downloads/MERRA2/"

    cons = nc.Dataset(f"{main_dir}MERRA2_100.const_2d_lnd_Nx.00000000.nc4")

    filenames = absoluteFilePaths("D:/Downloads/MERRA2/2021-06")
    filenames.extend(absoluteFilePaths("D:/Downloads/MERRA2/2021-07"))

    som_mean = np.zeros([1400, 2])
    som_baln = np.zeros([1400, 2])
    runoff = np.zeros(([1400, 2]))
    for id_monthly, filename in enumerate(filenames):
        gldas = nc.Dataset(filename)
        for id in np.arange(24):
            id_arr = int(id_monthly * 24 + id)
            if id_arr > 1399:
                continue
            som_mean[id_arr, 0] = float(filename[-8: -6]) + float(filename[-6: -4]) / 31 + id / 24 / 31
            som_baln[id_arr, 0] = float(filename[-8: -6]) + float(filename[-6: -4]) / 31 + id / 24 / 31
            # count points
            count = 0
            som = np.asarray(gldas["PRMC"][id, :, :]) * np.asarray(cons["dzpr"][0, :, :]) * 1000
            eva = np.asarray(gldas["EVLAND"][id, :, :]) * 60 * 60
            per = np.asarray(gldas["PRECTOTLAND"][id, :, :]) * 60 * 60 + np.asarray(gldas["PRECSNOLAND"][id, :, :]) * 60 * 60
            sur = np.asarray(gldas["RUNOFF"][id, :, :]) * 60 * 60
            for id_lat, lat in enumerate(lat_a):
                for id_lon, lon in enumerate(lon_a):
                    if np.logical_and(np.logical_and(lat >= lat_span[0], lat <= lat_span[1]),
                                      np.logical_and(lon >= lon_span[0], lon <= lon_span[1])):
                        if som[id_lat, id_lon] > -500.0:
                            count = count + 1
                            som_mean[id_arr, 1] = som_mean[id_arr, 1] + som[id_lat, id_lon]
                            som_baln[id_arr, 1] = som_baln[id_arr, 1] + per[id_lat, id_lon] - eva[id_lat, id_lon] \
                                                    - sur[id_lat, id_lon]
                            runoff[id_arr, 1] = runoff[id_arr, 1] + sur[id_lat, id_lon]

            som_mean[id_arr, 1] = som_mean[id_arr, 1] / count
            som_baln[id_arr, 1] = som_baln[id_arr, 1] / count
            runoff[id_arr, 1] = runoff[id_arr, 1] / count

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.step(som_mean[0: -1, 0], np.cumsum(np.diff(som_mean[:, 1])), label="soil_moisture_of_MERRA2", linewidth=2)
    ax.step(som_mean[:, 0], np.cumsum(som_baln[:, 1]), label="water_balance_of_MERRA2", linewidth=2)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.tick_params(labelsize=25, width=2.9)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel('kg/m$^2$', fontsize=20)
    ax.set_xlabel('Month of 2021', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    np.savetxt("../output/MERRA2_zhengzhou_soil.txt", som_mean)
    np.savetxt("../output/MERRA2_zhengzhou_soil_water_balance.txt", som_baln)
    np.savetxt("../output/MERRA2_zhengzhou_soil_runoff.txt", runoff)
    plt.show()


def check_discharge():
    coors_stations = np.asarray([[113.39, 34.5]])
    coors_stations = np.asarray([[112.24, 34.5], [115.05, 34.5]])
    coors_dams = np.asarray([[112.24, 34.5]])
    y = np.loadtxt("../input/yellow_river_subregion.txt")
    p = path.Path(y)
    num = 98

    areas = np.loadtxt("../input/areas_MERRA2.txt")
    areas[np.isnan(areas)] = 0
    discharges_measured = np.zeros([num, coors_stations.shape[0]])
    storage_measured = np.zeros([num, coors_dams.shape[0]])
    filenames_measured = os.listdir("D:/Downloads/dailyReportYellowRiver")
    for id, _ in enumerate(discharges_measured):
        tmp = np.asarray(pd.read_excel(f"D:/Downloads/dailyReportYellowRiver/{filenames_measured[id]}"))
        # discharges_measured[id, 0] = tmp[43, 4]
        discharges_measured[id, 0] = tmp[29, 4]
        discharges_measured[id, 1] = tmp[50, 4]
        storage_measured[id, 0] = tmp[28, 4][1: -2]

    # soil moisture in kg*m^-2
    lat_a = np.arange(-90., 90.5, 0.5)
    lon_a = np.arange(-180., 180., 0.625)

    main_dir = "D:/Downloads/MERRA2/"

    filenames = absoluteFilePaths("D:/Downloads/MERRA2/2021-06")
    filenames.extend(absoluteFilePaths("D:/Downloads/MERRA2/2021-07"))

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
            gldas = nc.Dataset(filename)
            sur = np.sum(gldas["RUNOFF"][:, :, :], axis=0) * is_in_polygon[id_sta, :, :]
            pre = np.sum(gldas["PRECTOTLAND"][:, :, :], axis=0) * is_in_polygon[id_sta, :, :]
            # fig, ax = plt.subplots(figsize=(12, 8))
            # plt.title(filename[-8: -4], fontsize=20)
            # cm = plt.cm.get_cmap("jet")
            # lat, lon = np.meshgrid(lat_a, lon_a)
            # im = ax.pcolormesh(lon, lat, pre.T, cmap=cm)
            # ax.plot(y[:, 0], y[:, 1])
            # ax.axis("equal")
            # ax.set_xlim([95, 120])
            # ax.set_ylim([30, 45])
            # cbar = fig.colorbar(im)
            # cbar.ax.set_title('kg m$^{-2}$ s$^{-1}$', fontsize=15)
            # plt.savefig(f"../image/background202106/MERRA2_precipitation_{filename[-8: -4]}.png")
            sur = sur * areas / 1000 / 24
            pre = pre * areas / 1000 / 24
            runoff[id_monthly, id_sta] = np.sum(sur)
            precip[id_monthly, id_sta] = np.sum(pre)

    gldas_precip = np.loadtxt("../output/GLDAS_precipitation.txt")
    gldas_runoff = np.loadtxt("../output/GLDAS_discharge.txt")

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    ax[0].plot(np.linspace(6, 9, num), runoff[:, 1] - runoff[:, 0], label="MERRA2_runoff", linewidth=2, marker="o")
    ax[0].plot(np.linspace(6, 9, gldas_runoff.__len__()), gldas_runoff[:, 1] - gldas_runoff[:, 0], label="GLDAS_runoff", linewidth=2, marker="o")
    ax[0].plot(np.linspace(6, 9, num), discharges_measured[:, 1] - discharges_measured[:, 0],
               label="measured_discharge", linewidth=2, marker="o")
    ax[0].yaxis.get_offset_text().set_fontsize(24)
    ax[0].tick_params(labelsize=25, width=2.9)
    ax[0].legend(fontsize=15, loc='best', frameon=False)
    ax[0].set_ylabel('Discharge [m$^3$/s]', fontsize=20)
    ax[0].set_xlabel('Month of 2021', fontsize=20)
    ax[0].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    ax[1].plot(np.linspace(6, 9, num), precip[:, 1] - precip[:, 0], label="MERRA2_precipitation", linewidth=2, marker="o")
    ax[1].plot(np.linspace(6, 9, gldas_precip.__len__()), gldas_precip[:, 1] - gldas_precip[:, 0], label="GLDAS_precipitation", linewidth=2, marker="o")
    ax[1].yaxis.get_offset_text().set_fontsize(24)
    ax[1].tick_params(labelsize=25, width=2.9)
    ax[1].legend(fontsize=15, loc='best', frameon=False)
    ax[1].set_ylabel('Precipitation [m$^3$/s]', fontsize=20)
    ax[1].set_xlabel('Month of 2021', fontsize=20)
    ax[1].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)

    ax[2].plot(np.linspace(6, 9, storage_measured.__len__()), storage_measured, label="storage_in_xiaolangdi", linewidth=2, marker="o")
    ax[2].yaxis.get_offset_text().set_fontsize(24)
    ax[2].tick_params(labelsize=25, width=2.9)
    ax[2].legend(fontsize=15, loc='best', frameon=False)
    ax[2].set_ylabel('Storage [10$^9$ m$^3$]', fontsize=20)
    ax[2].set_xlabel('Month of 2021', fontsize=20)
    ax[2].grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax[0].spines.values(), linewidth=3)
    plt.setp(ax[1].spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


def read_yellow_river():
    # with open("E:/lhsPrograms/daily_scripts/subregions.geojson") as f:
    #     gj = geojson.load(f)
    # for id in np.arange(gj["features"].__len__()):
    #     if gj["features"][id]["properties"]["SUBREGNAME"] == "YELLOW RIVER":
    #         np.savetxt("../input/yellow_river_subregion.txt",
    #                    gj["features"][id]["geometry"]["coordinates"][0][0],
    #                    delimiter=" ")

    y = np.loadtxt("../input/yellow_river_subregion.txt")
    p = Polygon(y, facecolor="k")
    fig, ax = plt.subplots()
    ax.plot(y[:, 0], y[:, 1])
    ax.axis("equal")
    plt.show()


def get_areas():
    # define wgs84 as crs
    geod = Geod('+a=6378137 +f=0.0033528106647475126')

    lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")
    areas = np.zeros([lat_a.__len__(), lon_a.__len__()])
    for id_lat, lat in enumerate(lat_a):
        for id_lon, lon in enumerate(lon_a):
            lat_small = lat - (lat_a[2] - lat_a[1]) / 2
            lat_large = lat + (lat_a[2] - lat_a[1]) / 2
            lon_small = lon - (lon_a[2] - lon_a[1]) / 2
            lon_large = lon + (lon_a[2] - lon_a[1]) / 2
            area, _ = geod.polygon_area_perimeter([lon_large, lon_large, lon_small, lon_small],
                                                  [lat_large, lat_small, lat_small, lat_large])
            areas[id_lat, id_lon] = np.abs(area)
    np.savetxt("../input/areas_GLDAS.txt", areas)


if __name__ == "__main__":
    # res = get_monthly_mean(2021, 6, "PRMC")
    # check_water_balance()
    # read_yellow_river()
    # get_areas()
    check_discharge()
