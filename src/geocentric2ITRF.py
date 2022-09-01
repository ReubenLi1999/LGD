from pyproj import Geod
import astropy.coordinates as ac
from astropy import units as u
from astropy.time import Time
import numpy as np


def main():
    source_locations = np.loadtxt("../output/coordinates_sources_syn.txt", dtype=np.float64, delimiter=",")

    # convert geocentric to itrs
    itrs = ac.ITRS(ac.WGS84GeodeticRepresentation(ac.Longitude(source_locations[:, 1], unit=u.degree),
                                                  ac.Latitude(source_locations[:, 0], unit=u.degree),
                                                  u.Quantity(source_locations[:, 2], unit=u.m)),
                   obstime=Time("2000-01-01T00:00:00"))
    q = ac.EarthLocation(*itrs.cartesian.xyz)

    np.savetxt("../output/coordinates_sources_syn_xyz.txt", np.c_[q.x.value, q.y.value, q.z.value])


if __name__ == "__main__":

    main()
