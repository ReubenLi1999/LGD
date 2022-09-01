import pygmt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom


def main():

    with open('d:/Downloads/china-geospatial-data-UTF8/CN-border-La.gmt') as src:
        context = src.read()
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]

    latlon = np.loadtxt("../input/2021-07-21/gracefo-c_GNV1B_2021-07-21_ground.txt", skiprows=6)

    # turn the lons and lats into a shapely LineString
    track = sgeom.LineString(zip(latlon[:, 0], latlon[:, 1]))
    #设置画图各种参数
    fig = plt.figure(figsize=[8, 8])
    # 设置投影类型和经纬度
    ax = plt.axes(projection=ccrs.LambertConformal(central_latitude=90,
                                                   central_longitude=105))
    # 画海，陆地，河流，湖泊
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    # 画国界
    for line in borders:
        ax.plot(line[0::2], line[1::2], '-', color='gray', transform=ccrs.Geodetic(), linewidth=0.7)
    # 画经纬度网格
    ax.gridlines(linestyle='--')
    # 框出区域
    ax.set_extent([80, 130, 13, 55])

    # 显示
    plt.show()


if __name__ == "__main__":
    main()
