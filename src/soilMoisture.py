import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():

    # ground track
    grd_c = np.loadtxt("../input/demo/grace-c_approximateOrbit_2021-07-22.txt", skiprows=4)
    grd_d = np.loadtxt("../input/demo/grace-d_approximateOrbit_2021-07-22.txt", skiprows=4)
    grd = (grd_d + grd_c) / 2.

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
    # rain precipitation
    rainf = np.loadtxt("../input/2021-07-20/Rainf_tavg_15.csv", delimiter=",")
    # difference
    som_diff = som_a - som_b
    # grid
    np.savetxt("../tmp/som_diff.csv", som_diff, delimiter=",")

    fig, ax = plt.subplots(figsize=(16, 8))
    cm = plt.cm.get_cmap("jet")
    plt.pcolormesh(Lon, Lat, som_diff.T, cmap=cm)
    ax.tick_params(labelsize=25, width=2.9)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.add_patch(Rectangle((110, 31), 6.5, 5.5, fill=None, alpha=1))
    ax.set_ylabel(r'Latitude [$\degree$]', fontsize=20)
    ax.set_xlabel(r'Longitude [$\degree$]', fontsize=20)
    ax.scatter(113.68, 34.75, color="k", marker="*")
    ax.axis('equal')
    # ax.set_title("2020-07-29", fontsize=24)
    plt.colorbar()
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


def rainf():
    # soil moisture grid
    lat = np.loadtxt("../input/2020-07-29/lat.csv", delimiter=",")
    lon = np.loadtxt("../input/2020-07-29/lon.csv", delimiter=",")
    Lat, Lon = np.meshgrid(lat, lon)

    date4plot = "20210720"
    hours = ["0000", "0300", "0600", "0900", "1200", "1500", "1800", "2100"]

    for id, hour in enumerate(hours):
        fn = nc.Dataset(f"D:/Downloads/gfo_dataset/GLDAS_NOAH025_3H.A{date4plot}.{hour}.021.nc4")
        fig, ax = plt.subplots(figsize=(16, 8))
        cm = plt.cm.get_cmap("jet")
        plt.pcolormesh(Lon, Lat, fn["Rainf_tavg"][0].T, cmap=cm)
        ax.tick_params(labelsize=25, width=2.9)
        ax.yaxis.get_offset_text().set_fontsize(24)
        # ax.legend(fontsize=15, loc='best', frameon=False)
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((110, 31), 6.5, 5.5, fill=None, alpha=1, color="r"))
        ax.set_ylabel(r'Latitude [$\degree$]', fontsize=20)
        ax.set_xlabel(r'Longitude [$\degree$]', fontsize=20)
        ax.scatter(113.68, 34.75, color="r", marker="*")
        ax.axis('equal')
        ax.set_xlim([110, 130])
        ax.set_ylim([25, 40])
        ax.set_title(f"{date4plot}:{hour}", fontsize=24)
        ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
        plt.colorbar(location="bottom", fraction=0.046, pad=0.2)
        plt.setp(ax.spines.values(), linewidth=3)
        plt.savefig(f"../image/rain_{date4plot}_{hour}.png", dpi=600)


if __name__ == "__main__":
    main()
    # rainf()
