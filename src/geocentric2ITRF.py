from pyproj import Geod
import astropy.coordinates as ac
from astropy import units as u
from astropy.time import Time
import numpy as np
import netCDF4 as nc


def main():
    main_dir = "D:/Downloads/MERRA2/"
    lats = np.loadtxt("../input/2020-07-02/lat.csv", delimiter=",")
    lons = np.loadtxt("../input/2020-07-02/lon.csv", delimiter=",")
    lons, lats = np.meshgrid(lons, lats)
    hgts = np.ones(lons.shape) * -1

    # convert geocentric to itrs
    itrs = ac.ITRS(ac.WGS84GeodeticRepresentation(ac.Longitude(lons, unit=u.degree),
                                                  ac.Latitude(lats, unit=u.degree),
                                                  u.Quantity(hgts, unit=u.m)),
                   obstime=Time("2000-01-01T00:00:00"))
    q = ac.EarthLocation(*itrs.cartesian.xyz)

    np.savetxt("../input/x_GLDAS_0.25x0.25_3.txt", q.x.value)
    np.savetxt("../input/y_GLDAS_0.25x0.25_3.txt", q.y.value)
    np.savetxt("../input/z_GLDAS_0.25x0.25_3.txt", q.z.value)


if __name__ == "__main__":

    main()
