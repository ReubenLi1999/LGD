import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy
import cartopy.crs as ccrs
import itertools
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import netCDF4 as nc
import datetime
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Polygon
from matplotlib import path
import matplotlib.patches
import matplotlib.colors as colors
import matplotlib.ticker as plticker
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit


def get_soil_moisture_diff(background, this):
    # soil moisture grid
    lat = np.loadtxt("../input/2020-07-29/lat.csv", delimiter=",")
    lon = np.loadtxt("../input/2020-07-29/lon.csv", delimiter=",")
    lat, lon = np.meshgrid(lat, lon)
    # soil moisture before the flood
    # monthly mean GLDAS dataset
    gldas_monthly_mean = nc.Dataset(f"D:/Downloads/GLDAS/monthly/GLDAS_NOAH025_M.A{background[-6:]}.021.nc4")
    gldas_daily_mean = nc.Dataset(
        f"D:/Downloads/GLDAS/three_hours/GLDAS_NOAH025_3H.A{this.strftime('%Y%m%d')}.0600.021.nc4")

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


def add_zebra_frame(ax, crs, lw=2, zorder=None):
    ax.spines["geo"].set_visible(False)
    left, right, bot, top = ax.get_extent()

    # Alternate black and white line segments
    bws = itertools.cycle(["k", "white"])

    xticks = sorted([left, *ax.get_xticks(), right])
    xticks = np.unique(np.array(xticks))
    yticks = sorted([bot, *ax.get_yticks(), top])
    yticks = np.unique(np.array(yticks))
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[bot, bot], [top, top]]
            else:
                xs = [[left, left], [right, right]]
                ys = [[start, end], [start, end]]

            # For first and lastlines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for (xx, yy) in zip(xs, ys):
                ax.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=lw,
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        pe.Stroke(linewidth=lw + 1, foreground="black"),
                        pe.Normal(),
                    ],
                )


def test4plot_tracks():
    sta_date = [2021, 7, 5]
    end_date = [2021, 8, 5]
    study_area = {"area": "zhengzhou", "lat_span": [33, 39], "lon_span": [113, 117], "lat_centre": 35}
    plot_area = {"extent": {"lat_span": [20, 50], "lon_span": [100, 130]},
                 "groundtrack": {"lat_span": [33, 37], "lon_span": [112, 118]}}
    border = np.loadtxt("../input/HeNan")
    # plot_tracks(sta_date, end_date, study_area, plot_area, step=5e-9, mode="all", border=border)

    sta_date = [2021, 6, 2]
    end_date = [2021, 8, 30]
    plot_precip_vs_lgd(sta_date, end_date, study_area, mode="all")

    sta_date = [2021, 7, 5]
    end_date = [2021, 8, 1]
    study_area = {"area": "europe", "lat_span": [45, 54], "lon_span": [0, 15], "lat_centre": 50}
    plot_area = {"extent": {"lat_span": [35, 70], "lon_span": [-10, 30]},
                 "groundtrack": {"lat_span": [47, 52], "lon_span": [2.5, 9.5]}}
    border = cartopy.feature.NaturalEarthFeature(category='cultural',
                                                 name='admin_0_boundary_lines_land', scale="50m", facecolor='none',
                                                 alpha=0.7)
    # plot_tracks(sta_date, end_date, study_area, plot_area, step=5e-9, mode="all", border=border)

    sta_date = [2021, 6, 2]
    end_date = [2021, 8, 30]
    plot_precip_vs_lgd(sta_date, end_date, study_area, mode="all")


def flip_legend(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def f_1(x, a, b, c, d, e, f, g, h):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f + g * x**6 + h * x**7


def plot_precip_vs_lgd(s_date, e_date, study_area, mode="all"):
    sta_date = datetime.date(s_date[0], s_date[1], s_date[2])
    end_date = datetime.date(e_date[0], e_date[1], e_date[2])
    dates4plot = [sta_date + datetime.timedelta(n) for n in range(int((end_date - sta_date).days))]
    dates4plot = [date_obj.strftime('%Y-%m-%d') for date_obj in dates4plot]

    lon_span = study_area["lon_span"]
    lat_span = study_area["lat_span"]
    area = study_area["area"]

    precip_merra = get_precipitation_in_area(dates4plot, [[lon_span[0], lat_span[0]], [lon_span[0], lat_span[1]],
                                                          [lon_span[1], lat_span[0]], [lon_span[1], lat_span[1]]], "MERRA-2")
    precip_gldas = get_precipitation_in_area(dates4plot, [[lon_span[0], lat_span[0]], [lon_span[0], lat_span[1]],
                                                          [lon_span[1], lat_span[0]], [lon_span[1], lat_span[1]]], "GLDAS")

    lgds2plot = load_lgd_by_study_area(dates4plot, area, mode)
    dates_plotted = list(lgds2plot["lri"].keys())

    # figure size
    fig = plt.figure(figsize=(14, 7))
    # the layout of subplots and their width ratio
    gs = gridspec.GridSpec(2, 1,
                           height_ratios=[1, 1],
                           hspace=0)
    # the LGDs for each track
    ax1 = plt.subplot(gs[0])
    # precipitation
    ax2 = plt.subplot(gs[1])

    ax1.plot(precip_merra[:, 0], precip_merra[:, 1], linewidth=2, marker="o", color="k", label="MERRA-2")
    ax1.plot(precip_gldas[:, 0], precip_gldas[:, 1], linewidth=2, marker="o", color="grey", label="GLDAS")
    tmp = np.c_[precip_merra[:, 1], precip_gldas[:, 1]]
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.tick_params(labelsize=15, width=2.9)
    ax1.legend(fontsize=13, loc="upper left", facecolor='white', framealpha=1)
    ax1.set_ylabel('Average Precipitation [mm]', fontsize=20)
    # ax1.set_xlabel('Day of 2021', fontsize=20)
    ax1.text(0.03, 0.1, "(a)", fontsize=15, bbox=dict(facecolor='grey', alpha=0.5),
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=1)
    plt.setp(ax1.spines.values(), linewidth=3)
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    ax1.xaxis.set_minor_locator(loc)
    ax1.yaxis.tick_right()
    ax1.set_ylim([np.min(tmp) - (np.max(tmp) - np.min(tmp)) / 10, np.max(tmp) + (np.max(tmp) - np.min(tmp)) / 10])
    ax1.tick_params(which='minor', axis="x", direction='in', length=8, width=3)
    ax1.tick_params(which='minor', axis="y", direction='in', length=8, width=3)
    ax1.tick_params(which='major', axis="x", direction='in', length=8)
    ax1.tick_params(which='major', axis="y", direction='in', length=8)
    ax1.yaxis.set_label_position("right")

    ave_lri = []
    ave_merra = []
    ave_gldas = []
    ave_kbr = []
    days_measr = []
    ave_measr = []
    days_model = []
    ave_model = []
    for _, date in enumerate(dates_plotted):
        day_of_year = datetime.datetime.strptime(f"2021-{date[0: 5]}", "%Y-%m-%d").timetuple().tm_yday
        days_model.append(day_of_year)
        days_model.append(day_of_year)
        days_measr.append(day_of_year)
        days_measr.append(day_of_year)
        mask_new = np.argmin(np.abs(lgds2plot["lat"][date] - study_area["lat_centre"]))
        if np.max(np.abs(lgds2plot["lri"][date])) > 1e-9:
            ave_lri.append(np.mean(lgds2plot["lri"][date][mask_new]))
            ave_measr.append(np.mean(lgds2plot["lri"][date][mask_new]))
        else:
            ave_lri.append(np.nan)
            ave_measr.append(np.nan)
        if np.abs(np.mean(lgds2plot["kbr"][date][mask_new]) - np.mean(lgds2plot["GLDAS"][date][mask_new])) < 3e-9:
            ave_kbr.append(np.mean(lgds2plot["kbr"][date][mask_new]))
            ave_measr.append(np.mean(lgds2plot["kbr"][date][mask_new]))
        else:
            ave_kbr.append(np.nan)
            ave_measr.append(np.nan)

        ave_gldas.append(np.mean(lgds2plot["GLDAS"][date][mask_new]))
        ave_merra.append(np.mean(lgds2plot["MERRA2"][date][mask_new]))
        ave_model.append(np.mean(lgds2plot["GLDAS"][date][mask_new]))
        ave_model.append(np.mean(lgds2plot["MERRA2"][date][mask_new]))

    ave_lri = np.asarray(ave_lri) * 1e9
    ave_merra = np.asarray(ave_merra) * 1e9
    ave_gldas = np.asarray(ave_gldas) * 1e9
    ave_kbr = np.asarray(ave_kbr) * 1e9
    ave_measr = np.asarray(ave_measr) * 1e9
    ave_model = np.asarray(ave_model) * 1e9
    day_of_year_arr = []
    for id, date in enumerate(dates_plotted):
        date = date[0: 5]
        day_of_year = datetime.datetime.strptime(f"2021-{date}", "%Y-%m-%d").timetuple().tm_yday
        day_of_year_arr.append(day_of_year)

    day_of_year_arr = np.asarray(day_of_year_arr)
    ax2.scatter(day_of_year_arr, ave_lri, s=70, marker="o", color="red", label="LRI")
    ax2.scatter(day_of_year_arr, ave_kbr, s=70, marker="v", color="red", alpha=0.3, label="KBR")

    days_measr = np.asarray(days_measr)
    days_model = np.asarray(days_model)
    popt, pcov = curve_fit(f_1, days_measr[~np.isnan(ave_measr)], ave_measr[~np.isnan(ave_measr)])
    ax2.plot(precip_merra[:, 0], f_1(precip_merra[:, 0], *popt), '-', color="r", alpha=0.5, linewidth=2, label="fit_modelled")

    ax2.scatter(day_of_year_arr, ave_gldas, s=70, marker="*", color="blue", label="GLDAS")
    ax2.scatter(day_of_year_arr, ave_merra, s=70, marker="x", color="blue", alpha=0.3, label="MERRA-2")

    popt, pcov = curve_fit(f_1, days_model, ave_model)
    ax2.plot(precip_merra[:, 0], f_1(precip_merra[:, 0], *popt), '-', color="b", alpha=0.5, linewidth=2, label="fit_measured")
    tmp = np.c_[ave_lri, ave_kbr, ave_merra, ave_gldas]
    tmp = tmp[~np.isnan(tmp)]
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(flip_legend(handles, 3), flip_legend(labels, 3), fontsize=13,
               loc="upper left", facecolor='white', framealpha=1, ncol=3)
    ax2.yaxis.get_offset_text().set_fontsize(18)
    ax2.tick_params(labelsize=15, width=2.9)
    if study_area["lat_centre"] > 0:
        ax2.set_ylabel(rf'LGD at {study_area["lat_centre"]} °N [nm/s$^2$]', fontsize=20)
    else:
        ax2.set_ylabel(rf'LGD at {study_area["lat_centre"]} °S [nm/s$^2$]', fontsize=20)
    ax2.set_xlabel('Day of 2021', fontsize=20)
    ax2.text(0.03, 0.1, "(b)", fontsize=15, bbox=dict(facecolor='grey', alpha=0.5),
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=1)
    plt.setp(ax2.spines.values(), linewidth=3)
    ax2.set_xlim(ax1.get_xlim())
    # the second x ticks, the date for each track
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    ax2.xaxis.set_minor_locator(loc)
    ax2.tick_params(which='minor', axis="x", direction='in', length=8, width=3)
    ax2.tick_params(which='minor', axis="y", direction='in', length=8, width=3)
    ax2.tick_params(which='major', axis="x", direction='in', length=8)
    ax2.tick_params(which='major', axis="y", direction='in', length=8)
    ax2.tick_params(axis="x", direction='in', length=8)
    ax2.tick_params(axis="y", direction='in', length=8)
    ax2.set_ylim([np.min(tmp) - 0.5, np.max(tmp) + 2.5])
    plt.savefig(f"../image/{area}_precip_vs_lgd.png", dpi=600, bbox_inches='tight')


def plot_tracks(s_date, e_date, study_area, plot_area, step, mode, xlim=None, border=None):
    # date array
    if xlim is None:
        xlim = [0, 0]
    sta_date = datetime.date(s_date[0], s_date[1], s_date[2])
    end_date = datetime.date(e_date[0], e_date[1], e_date[2])
    dates4plot = [sta_date + datetime.timedelta(n) for n in range(int((end_date - sta_date).days))]
    dates4plot = [date_obj.strftime('%Y-%m-%d') for date_obj in dates4plot]

    lon_span = study_area["lon_span"]
    lat_span = study_area["lat_span"]
    area = study_area["area"]

    precip = get_precipitation_in_area(dates4plot, [[lon_span[0], lat_span[0]], [lon_span[0], lat_span[1]],
                                                    [lon_span[1], lat_span[0]], [lon_span[1], lat_span[1]]])
    precip_gldas = get_precipitation_in_area(dates4plot, [[lon_span[0], lat_span[0]], [lon_span[0], lat_span[1]],
                                                          [lon_span[1], lat_span[0]], [lon_span[1], lat_span[1]]], "GLDAS")

    lgds2plot = load_lgd_by_study_area(dates4plot, area, mode)
    dates4plot = list(lgds2plot["lri"].keys())

    # figure size
    fig = plt.figure(figsize=(20, 10))
    # main title for this figure
    # fig.suptitle("GRACE Follow-On tracks and the corresponding LGDs", fontsize=24, style='italic')
    # the layout of subplots and their width ratio
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[8, 20],
                           height_ratios=[1, 1],
                           wspace=0.05,
                           hspace=0.40)
    # the map subplot and its projection
    ax1 = plt.subplot(gs[2], projection=ccrs.PlateCarree())
    # the LGDs for each track
    ax2 = plt.subplot(gs[3])
    # precipitation
    ax3 = plt.subplot(gs[1])
    # average amplitude of LGDs
    # ax4 = plt.subplot(gs[5])
    # plot LGDs for each track
    count = 0
    dates_plotted = []
    ave_lri = []
    ave_merra = []
    ave_gldas = []
    ave_kbr = []
    for date in lgds2plot["lri"]:
        mask = np.where(np.logical_and(lgds2plot["lat"][date] < plot_area["groundtrack"]["lat_span"][1],
                                       lgds2plot["lat"][date] > plot_area["groundtrack"]["lat_span"][0]))[0]
        if np.logical_and(np.mean(lgds2plot["lon"][date][mask]) > plot_area["groundtrack"]["lon_span"][0],
                          np.mean(lgds2plot["lon"][date][mask]) < plot_area["groundtrack"]["lon_span"][1]):
            ax2.plot(lgds2plot["lat"][date], lgds2plot["GLDAS"][date] + step * count,
                     label="GLDAS", linewidth=2, color="green")
            ax2.plot(lgds2plot["lat"][date], lgds2plot["MERRA2"][date] + step * count,
                     label="MERRA-2", linewidth=2, color="blue")
            if np.max(np.abs(lgds2plot["lri"][date])) > 1e-9:
                ax2.plot(lgds2plot["lat"][date], lgds2plot["lri"][date] + step * count,
                         label=f"LRI", linewidth=2, color="red", linestyle="dashed")
            else:
                ax2.plot(np.nan, np.nan,
                         label=f"LRI", linewidth=2, color="red", linestyle="dashed")
            ax2.plot(lgds2plot["lat"][date], lgds2plot["kbr"][date] + step * count,
                     label=f"KBR", linewidth=4, color="red", alpha=0.3)
            count = count + 1
            dates_plotted.append(date)
            mask_new = np.argmin(np.abs(lgds2plot["lat"][date] - study_area["lat_centre"]))
            if np.max(np.abs(lgds2plot["lri"][date])) > 1e-9:
                ave_lri.append(np.mean(lgds2plot["lri"][date][mask_new]))
            else:
                ave_lri.append(np.nan)
            ave_merra.append(np.mean(lgds2plot["MERRA2"][date][mask_new]))
            ave_gldas.append(np.mean(lgds2plot["GLDAS"][date][mask_new]))
            ave_kbr.append(np.mean(lgds2plot["kbr"][date][mask_new]))
    # switch the x axis and the y axis
    for line in ax2.lines:
        # get data from first line of the plot
        new_x = line.get_ydata()
        new_y = line.get_xdata()
        # set new x- and y- data for the line
        line.set_xdata(new_x)
        line.set_ydata(new_y)
    # add one x axis to show the date of each track
    ax22 = ax2.twiny()
    ax2.xaxis.get_offset_text().set_fontsize(14)
    # highlight the study area
    ax2.axhspan(lat_span[0], lat_span[1], alpha=0.5, color='grey')
    ax2.tick_params(labelsize=15, width=2.9)
    ax2.legend(("GLDAS", "MERRA-2", "LRI", "KBR"), fontsize=15,
               loc="upper left", facecolor='white', framealpha=1, ncol=2)
    ax2.set_xlabel('LGD [nm/s$^2$]', fontsize=20)
    xlim = [-step, step * count]
    ax2.set_xlim(xlim)
    ax2.yaxis.tick_right()
    ax2.set_ylim(plot_area["extent"]["lat_span"])
    # label the subplot
    ax2.text(0.03, 0.1, "(c)", fontsize=15, bbox=dict(facecolor='grey', alpha=0.5),
             horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    ax2.grid(True, which='both', ls='dashed', color='0.5', linewidth=1)
    # the x ticks for this subplot
    ax_labels = [f"{s:01}" for s in np.asarray(np.arange(0, dates_plotted.__len__()) * step * 1e9, dtype=int)]
    # the first x ticks, the amplitude of LGDs
    ax2.set_xticks(np.arange(0, dates_plotted.__len__()) * step, ax_labels, rotation=45, ha='right',
                   rotation_mode='anchor')
    ax22.set_xlim(ax2.get_xlim())
    # the second x ticks, the date for each track
    ax2_xtick = np.arange(0, dates_plotted.__len__(), dtype=np.float64) * step
    ax22.set_xticks(ax2_xtick)
    ax22.set_xticklabels(dates_plotted, rotation=45, fontsize=15)
    # thicken the axis bar
    plt.setp(ax2.spines.values(), linewidth=3)
    plt.setp(ax1.spines.values(), linewidth=3)

    # define colourmap
    cm = plt.cm.get_cmap("RdBu_r")
    lat, lon, som_diff = get_soil_moisture_diff("background202106", datetime.datetime(2021, 7, 21))
    # contour the map, projection and normalisation are needed
    im = ax1.pcolormesh(lon, lat, som_diff.T, cmap=cm, transform=ccrs.PlateCarree(),
                        norm=colors.CenteredNorm())

    # Setup zoom window
    scale = (plot_area["extent"]["lon_span"][1] - plot_area["extent"]["lon_span"][0]) / (
                lon_span[1] - lon_span[0]) * 0.9
    if scale > 3.5:
        scale = 3.5
    axins1 = zoomed_inset_axes(ax1, scale,
                               loc='upper left', bbox_to_anchor=(0., 2.3), bbox_transform=ax1.transAxes)
    mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="0.5")
    axins1.set_xlim(study_area["lon_span"])
    axins1.set_ylim(study_area["lat_span"])
    axins1.tick_params(labelsize=25, width=2.9)
    axins1.tick_params(which='major', axis="x", direction='in', length=8)
    axins1.tick_params(which='major', axis="y", direction='in', length=8)
    axins1.pcolormesh(lon, lat, som_diff.T, cmap=cm, norm=colors.CenteredNorm())

    # plot the ground tracks
    for date in lgds2plot["lri"]:
        mask = np.where(np.logical_and(lgds2plot["lat"][date] < plot_area["groundtrack"]["lat_span"][1],
                                       lgds2plot["lat"][date] > plot_area["groundtrack"]["lat_span"][0]))[0]
        if np.logical_and(np.mean(lgds2plot["lon"][date][mask]) > plot_area["groundtrack"]["lon_span"][0],
                          np.mean(lgds2plot["lon"][date][mask]) < plot_area["groundtrack"]["lon_span"][1]):
            if "#" in date:
                ax1.plot(lgds2plot["lon"][date], lgds2plot["lat"][date],
                         color="purple", linewidth=1, transform=ccrs.PlateCarree())
            else:
                ax1.plot(lgds2plot["lon"][date], lgds2plot["lat"][date], alpha=0.5,
                         color="purple", linewidth=1, transform=ccrs.PlateCarree())
    ax1.tick_params(labelsize=25, width=2.9)
    ax1.yaxis.get_offset_text().set_fontsize(24)
    ax1.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    ax1.coastlines()
    ax1.set_extent([plot_area["extent"]["lon_span"][0], plot_area["extent"]["lon_span"][1],
                    plot_area["extent"]["lat_span"][0], plot_area["extent"]["lat_span"][1]],
                   crs=ccrs.PlateCarree())
    ax1.set_xticks(np.arange(plot_area["extent"]["lon_span"][0], plot_area["extent"]["lon_span"][1] + 1, 10))
    ax1.set_yticks(np.arange(plot_area["extent"]["lat_span"][0], plot_area["extent"]["lat_span"][1] + 1, 10))
    ax1.set_ylabel('Lat. [deg]', fontsize=20, labelpad=45)
    ax1.set_xlabel('Lon. [deg]', fontsize=20, labelpad=20)
    # draw the border
    if type(border) == cartopy.feature.NaturalEarthFeature:
        ax1.add_feature(border, linestyle='--', edgecolor='k', alpha=1)
    else:
        ax1.plot(border[:, 0], border[:, 1], transform=ccrs.PlateCarree(), color="k", linewidth=2, linestyle="--")

    ax1.text(0.1, 0.1, "(b)", fontsize=15, bbox=dict(facecolor='grey', alpha=0.5),
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=2, color='gray', alpha=0.5, linestyle='--',
                       xlocs=np.arange(plot_area["extent"]["lon_span"][0], plot_area["extent"]["lon_span"][1] + 1, 10),
                       ylocs=np.arange(plot_area["extent"]["lat_span"][0], plot_area["extent"]["lat_span"][1] + 1, 10))
    gl.xlabel_style = {"fontsize": 15}
    gl.ylabel_style = {"fontsize": 15}
    add_zebra_frame(ax1, crs=ccrs.PlateCarree())
    ax1.set_xticks([])
    ax1.set_yticks([])
    axins = inset_axes(ax1,
                       width="50%",  # width = 5% of parent_bbox width
                       height="5%",  # height : 50%
                       loc='upper left',
                       bbox_to_anchor=(0, 0.3, 2, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )
    cbar = fig.colorbar(im, cax=axins, aspect=1, shrink=0.5, orientation="horizontal")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_ticks([-200, -100, 0, 100, 200])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.set_title('EWH [mm]', fontsize=15)

    ax3.plot(precip[:, 0], precip[:, 1], linewidth=2, marker="o", color="k", label="MERRA-2")
    ax3.plot(precip_gldas[:, 0], precip_gldas[:, 1], linewidth=2, marker="o", color="grey", label="GLDAS")
    tmp = np.c_[precip[:, 1], precip_gldas[:, 1]]
    ax3.yaxis.get_offset_text().set_fontsize(14)
    ax3.tick_params(labelsize=15, width=2.9)
    ax3.legend(fontsize=15, loc="upper left", facecolor='white', framealpha=1)
    ax3.set_ylabel('Average Precipitation [mm]', fontsize=20)
    ax3.set_xlabel('Day of 2021', fontsize=20)
    ax3.text(0.03, 0.1, "(a)", fontsize=15, bbox=dict(facecolor='grey', alpha=0.5),
             horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.grid(True, which='both', ls='dashed', color='0.5', linewidth=1)
    plt.setp(ax3.spines.values(), linewidth=3)
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    ax3.xaxis.set_minor_locator(loc)
    ax3.set_ylim([np.min(tmp) - (np.max(tmp) - np.min(tmp)) / 10, np.max(tmp) + (np.max(tmp) - np.min(tmp)) / 10])
    ax3.tick_params(which='minor', axis="x", direction='in', length=8, width=3)
    ax3.tick_params(which='minor', axis="y", direction='in', length=8, width=3)
    ax3.tick_params(which='major', axis="x", direction='in', length=8)
    ax3.tick_params(which='major', axis="y", direction='in', length=8)

    # Create the arrow
    for id, date in enumerate(dates_plotted):
        date = date[0: 5]
        if any(date in date_plotted for date_plotted in dates_plotted):
            day_of_year = datetime.datetime.strptime(f"2021-{date}", "%Y-%m-%d").timetuple().tm_yday
            arrow = matplotlib.patches.ConnectionPatch(
                [day_of_year, ax3.get_ylim()[0]],
                [ax2_xtick[id], ax2.get_ylim()[1] + 6],
                coordsA=ax3.transData,
                coordsB=ax2.transData,
                # Default shrink parameter is 0 so can be omitted
                color="orange",
                arrowstyle="-|>",  # "normal" arrow
                mutation_scale=30,  # controls arrow head size
                linewidth=3,
                fc="orange",
                alpha=0.8
            )
            # 5. Add patch to list of objects to draw onto the figure
            fig.patches.append(arrow)

    # Create the arrow
    # 1. Get transformation operators for axis and figure
    # ax0tr = ax2.transData  # Axis 0 -> Display
    # ax1tr = ax4.transData  # Axis 1 -> Display
    # figtr = fig.transFigure.inverted()  # Display -> Figure
    # # 2. Transform arrow start point from axis 0 to figure coordinates
    # pt_b = figtr.transform(ax0tr.transform((np.max(ax2_xtick) + step, np.mean(lat_span))))
    # # 3. Transform arrow end point from axis 1 to figure coordinates
    # pt_e = figtr.transform(ax1tr.transform((ax4.get_xlim()[1], np.mean(ax4.get_ylim()))))
    # # 4. Create the patch
    # arrow = matplotlib.patches.FancyArrowPatch(
    #     pt_b, pt_e, transform=fig.transFigure,  # Place arrow in figure coord system
    #     fc="grey", connectionstyle="arc3,rad=-0.4", arrowstyle='simple', alpha=0.3,
    #     mutation_scale=40.
    # )
    # # 5. Add patch to list of objects to draw onto the figure
    # fig.patches.append(arrow)

    plt.savefig(f"../image/{area}.png", bbox_inches='tight', dpi=600)
    plt.savefig(f"../image/{area}.pdf", bbox_inches='tight', dpi=600)


def test4check_consistence():
    sta_date = [2021, 6, 2]
    end_date = [2021, 8, 31]
    # study area
    study_area = {"area": "zhengzhou", "lat_span": [33, 37], "lon_span": [113, 117]}
    plot_area = {"lat_span": [20, 60], "lon_span": [90, 140]}

    check_consistence(sta_date, end_date, study_area, plot_area)


def check_consistence(sta_date, end_date, study_area, plot_area):
    # date array
    sta_date = datetime.date(sta_date[0], sta_date[1], sta_date[2])
    end_date = datetime.date(end_date[0], end_date[1], end_date[2])
    dates4plot = [sta_date + datetime.timedelta(n) for n in range(int((end_date - sta_date).days))]
    dates4plot = [date_obj.strftime('%Y-%m-%d') for date_obj in dates4plot]

    lgds4plot = load_lgd_by_study_area(dates4plot, study_area["area"], "all")

    date_arr = np.arange(0, lgds4plot["lri"].__len__(), 1)
    rms_lri_vs_gldas = np.zeros([lgds4plot["lri"].__len__(), 1])
    rms_lri_vs_merra = np.zeros([lgds4plot["lri"].__len__(), 1])
    rms_kbr_vs_gldas = np.zeros([lgds4plot["lri"].__len__(), 1])
    rms_kbr_vs_merra = np.zeros([lgds4plot["lri"].__len__(), 1])
    count = 0
    for key in lgds4plot["lri"]:
        mask = np.where(np.logical_and(lgds4plot["lat"][key] > plot_area["lat_span"][0],
                                       lgds4plot["lat"][key] < plot_area["lat_span"][1]))[0]
        rms_lri_vs_gldas[count] = rmse(lgds4plot["lri"][key][mask], lgds4plot["GLDAS"][key][mask])
        rms_lri_vs_merra[count] = rmse(lgds4plot["lri"][key][mask], lgds4plot["MERRA2"][key][mask])
        rms_kbr_vs_gldas[count] = rmse(lgds4plot["kbr"][key][mask], lgds4plot["GLDAS"][key][mask])
        rms_kbr_vs_merra[count] = rmse(lgds4plot["kbr"][key][mask], lgds4plot["MERRA2"][key][mask])
        count = count + 1

    fig, ax = plt.subplots(figsize=(20, 8))
    fig.suptitle(f"Root mean square error", fontsize=24)
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.plot(date_arr, rms_lri_vs_gldas - 5e-9, label="LRI vs GLDAS", marker="o")
    ax.plot(date_arr, rms_lri_vs_merra - 5e-9, label="LRI vs MERRA2", marker="*")
    ax.plot(date_arr, rms_kbr_vs_gldas, label="KBR vs GLDAS", marker="v")
    ax.plot(date_arr, rms_kbr_vs_merra, label="KBR vs MERRA2", marker="x")
    ax.tick_params(labelsize=25, width=2.9)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.set_ylabel('RMS', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


def load_lgd_by_study_area(dates4plot, study_area, mode):
    files_track = os.listdir(f"../output/{study_area}")
    lgds_lri = {}
    lgds_kbr = {}
    lgds_gldas = {}
    lgds_merra = {}
    lgds_lri_ascend = {}
    lgds_kbr_ascend = {}
    lgds_lri_descend = {}
    lgds_kbr_descend = {}
    lgds_gldas_ascend = {}
    lgds_merra_ascend = {}
    lgds_gldas_descend = {}
    lgds_merra_descend = {}
    lat = {}
    lon = {}
    lat_ascend = {}
    lon_ascend = {}
    lat_descend = {}
    lon_descend = {}
    dates_ascend = []
    dates_descend = []
    for _, filename in enumerate(files_track):
        tmp = np.loadtxt(f"../output/{study_area}/{filename}")
        if "KBR" in filename:
            if tmp[20, 1] - tmp[10, 1] > 0:
                lgds_kbr_ascend[filename[5: 10]] = tmp[10: -10, :]
                lgds_kbr[f"{filename[5: 10]}#"] = tmp[10: -10, :]
                dates_ascend.append(filename[5: 10])
            else:
                lgds_kbr_descend[filename[5: 10]] = tmp[10: -10, :]
                lgds_kbr[f"{filename[5: 10]}"] = tmp[10: -10, :]
                dates_descend.append(filename[5: 10])
        else:
            if (tmp[:, 2] > 1e-8).any():
                tmp[:, 2] = np.nan
            if tmp[20, 1] - tmp[10, 1] > 0:
                lgds_lri_ascend[filename[5: 10]] = tmp[10: -10, 2]
                lgds_gldas_ascend[filename[5: 10]] = tmp[10: -10, 4]
                lgds_merra_ascend[filename[5: 10]] = tmp[10: -10, 3]
                lon_ascend[filename[5: 10]] = tmp[10: -10, 0]
                lat_ascend[filename[5: 10]] = tmp[10: -10, 1]
                lgds_lri[f"{filename[5: 10]}#"] = tmp[10: -10, 2]
                lgds_gldas[f"{filename[5: 10]}#"] = tmp[10: -10, 4]
                lgds_merra[f"{filename[5: 10]}#"] = tmp[10: -10, 3]
                lon[f"{filename[5: 10]}#"] = tmp[10: -10, 0]
                lat[f"{filename[5: 10]}#"] = tmp[10: -10, 1]
            else:
                lgds_lri_descend[filename[5: 10]] = tmp[10: -10, 2]
                lgds_gldas_descend[filename[5: 10]] = tmp[10: -10, 4]
                lgds_merra_descend[filename[5: 10]] = tmp[10: -10, 3]
                lon_descend[filename[5: 10]] = tmp[10: -10, 0]
                lat_descend[filename[5: 10]] = tmp[10: -10, 1]
                lgds_lri[f"{filename[5: 10]}"] = tmp[10: -10, 2]
                lgds_gldas[f"{filename[5: 10]}"] = tmp[10: -10, 4]
                lgds_merra[f"{filename[5: 10]}"] = tmp[10: -10, 3]
                lon[f"{filename[5: 10]}"] = tmp[10: -10, 0]
                lat[f"{filename[5: 10]}"] = tmp[10: -10, 1]

    lgds_ascend = {"lon": lon_ascend,
                   "lat": lat_ascend,
                   "lri": lgds_lri_ascend,
                   "GLDAS": lgds_gldas_ascend,
                   "MERRA2": lgds_merra_ascend,
                   "kbr": {}}

    del_keys = []
    for key in lgds_lri_ascend:
        f = interpolate.interp1d(lgds_kbr_ascend[key][:, 1], lgds_kbr_ascend[key][:, 2], fill_value='extrapolate')
        lgds_ascend["kbr"][key] = f(lgds_ascend["lat"][key])
        if not any(key[0: 5] in date for date in dates4plot):
            del_keys.append(key)
    for _, key in enumerate(del_keys):
        del lgds_ascend["kbr"][key]
        del lgds_ascend["lri"][key]
        del lgds_ascend["GLDAS"][key]
        del lgds_ascend["MERRA2"][key]

    lgds_descend = {"lon": lon_descend,
                    "lat": lat_descend,
                    "lri": lgds_lri_descend,
                    "GLDAS": lgds_gldas_descend,
                    "MERRA2": lgds_merra_descend,
                    "kbr": {}}

    del_keys = []
    for key in lgds_lri_descend:
        f = interpolate.interp1d(lgds_kbr_descend[key][:, 1], lgds_kbr_descend[key][:, 2], fill_value='extrapolate')
        lgds_descend["kbr"][key] = f(lgds_descend["lat"][key])
        if not any(key[0: 5] in date for date in dates4plot):
            del_keys.append(key)
    for _, key in enumerate(del_keys):
        del lgds_descend["kbr"][key]
        del lgds_descend["lri"][key]
        del lgds_descend["GLDAS"][key]
        del lgds_descend["MERRA2"][key]

    lgds = {"lon": lon,
            "lat": lat,
            "lri": lgds_lri,
            "GLDAS": lgds_gldas,
            "MERRA2": lgds_merra,
            "kbr": {}}

    del_keys = []
    for key in lgds_lri:
        f = interpolate.interp1d(lgds_kbr[key][:, 1], lgds_kbr[key][:, 2], fill_value='extrapolate')
        lgds["kbr"][key] = f(lgds["lat"][key])
        if not any(key[0: 5] in date for date in dates4plot):
            del_keys.append(key)
    for _, key in enumerate(del_keys):
        del lgds["kbr"][key]
        del lgds["lri"][key]
        del lgds["GLDAS"][key]
        del lgds["MERRA2"][key]

    if mode == "ascend":
        lgds2plot = lgds_ascend
    elif mode == "descend":
        lgds2plot = lgds_descend
    else:
        lgds2plot = lgds

    return lgds2plot


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def get_precipitation_in_area(dates4pre, y, hydro_model=None):
    if hydro_model is None:
        hydro_model = "MERRA-2"

    p = path.Path(y)

    if hydro_model == "MERRA-2":
        areas = np.loadtxt("../input/areas_MERRA2.txt")
        areas[np.isnan(areas)] = 0
        # soil moisture in kg*m^-2
        lat_a = np.arange(-90., 90.5, 0.5)
        lon_a = np.arange(-180., 180., 0.625)

        filenames = absolute_file_paths("D:/Downloads/MERRA2/2021-06")
        filenames.extend(absolute_file_paths("D:/Downloads/MERRA2/2021-07"))
        filenames.extend(absolute_file_paths("D:/Downloads/MERRA2/2021-08"))

        id_arr = []
        time_arr = []
        for id, filename in enumerate(filenames):
            if any(f"{date[0: 4]}{date[5: 7]}{date[8: 10]}" in filename for date in dates4pre):
                id_arr.append(filename)

        for id, date in enumerate(dates4pre):
            time_arr.append(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday)
        filenames = id_arr

        precip = np.zeros([filenames.__len__(), 1])
        is_in_polygon = np.zeros([lat_a.__len__(), lon_a.__len__()])
        for id_lat, lat in enumerate(lat_a):
            for id_lon, lon in enumerate(lon_a):
                if p.contains_points([[lon, lat]])[0]:
                    is_in_polygon[id_lat, id_lon] = 1

        for id_monthly, filename in enumerate(filenames):
            gldas = nc.Dataset(filename)
            pre = np.sum(gldas["PRECTOTLAND"][:, :, :] * 3600, axis=0) * is_in_polygon * areas
            precip[id_monthly] = np.sum(pre) / np.sum(is_in_polygon * areas)
    else:
        areas = np.loadtxt("../input/areas_GLDAS.txt")
        areas[np.isnan(areas)] = 0
        # soil moisture in kg*m^-2
        lat_a = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
        lon_a = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")

        filenames = absolute_file_paths("D:/Downloads/GLDAS/three_hours")
        id_arr = []
        time_arr = []
        for id, filename in enumerate(filenames):
            if any(f"{date[0: 4]}{date[5: 7]}{date[8: 10]}" in filename for date in dates4pre):
                id_arr.append(filename)

        for id, date in enumerate(dates4pre):
            time_arr.append(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday)
        filenames = id_arr

        precip = np.zeros([filenames.__len__(), 1])
        is_in_polygon = np.zeros([lat_a.__len__(), lon_a.__len__()])
        for id_lat, lat in enumerate(lat_a):
            for id_lon, lon in enumerate(lon_a):
                if p.contains_points([[lon, lat]])[0]:
                    is_in_polygon[id_lat, id_lon] = 1

        for id_monthly, filename in enumerate(filenames):
            gldas = nc.Dataset(filename)
            pre = np.asarray(gldas["Rainf_f_tavg"][0])
            pre[pre == -9999] = 0
            pre = pre * is_in_polygon * areas
            precip[id_monthly] = np.sum(pre) / np.sum(is_in_polygon * areas) * 3600 * 3

        precip = np.add.reduceat(precip, np.arange(0, len(precip), 8))
    return np.c_[time_arr, precip]


def absolute_file_paths(directory):
    res = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            res.append(os.path.abspath(os.path.join(dirpath, f)))

    return res


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    test4plot_tracks()
    # test4check_consistence()
