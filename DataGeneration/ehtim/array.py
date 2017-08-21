from builtins import str
from builtins import range
from builtins import object

import numpy as np

import ehtim.observing.obs_simulate as simobs
import ehtim.io.save
import ehtim.io.load

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

###########################################################################################################################################
#Array object
###########################################################################################################################################
class Array(object):
    """A VLBI array of telescopes with site locations, SEFDs, and other data.

       Attributes:
           tarr (numpy.recarray): The array of telescope data with datatype DTARR
           tkey (dict): A dictionary of rows in the tarr for each site name
           ephem (dict): A dictionary of 2TLEs for each space antenna, Space antennas have x=y=z=0 in the tarr
    """

    def __init__(self, tarr, ephem={}):
        self.tarr = tarr
        self.ephem = ephem

        # check to see if ephemeris is correct
        for line in self.tarr:
            if np.any(np.isnan([line['x'],line['y'],line['z']])):
                sitename = str(line['site'])
                try:
                    elen = len(ephem[sitename])
                except NameError:
                    raise Exception ('no ephemeris for site %s !' % sitename)
                if elen != 3:

                    raise Exception ('wrong ephemeris format for site %s !' % sitename)

        # Dictionary of array indices for site names
        self.tkey = {self.tarr[i]['site']: i for i in range(len(self.tarr))}

    def listbls(self):
        """List all baselines.
        """

        bls = []
        for i1 in sorted(self.tarr['site']):
            for i2 in sorted(self.tarr['site']):
                if not ([i1,i2] in bls) and not ([i2,i1] in bls) and i1 != i2:
                    bls.append([i1,i2])

        return np.array(bls)

    def obsdata(self, ra, dec, rf, bw, tint, tadv, tstart, tstop, mjd=MJD_DEFAULT,
                      timetype='UTC', elevmin=ELEV_LOW, elevmax=ELEV_HIGH, tau=TAUDEF):

        """Generate u,v points and baseline uncertainties.

           Args:
               ra (float): the source right ascension in fractional hours
               dec (float): the source declination in fractional degrees
               tint (float): the scan integration time in seconds
               tadv (float): the uniform cadence between scans in seconds
               tstart (float): the start time of the observation in hours
               tstop (float): the end time of the observation in hours
               mjd (int): the mjd of the observation
               timetype (str): how to interpret tstart and tstop; either 'GMST' or 'UTC'
               elevmin (float): station minimum elevation in degrees
               elevmax (float): station maximum elevation in degrees
               tau (float): the base opacity at all sites, or a dict giving one opacity per site

           Returns:
               Obsdata: an observation object with no data

        """

        obsarr = simobs.make_uvpoints(self, ra, dec, rf, bw,
                                            tint, tadv, tstart, tstop,
                                            mjd=mjd, tau=tau,
                                            elevmin=elevmin, elevmax=elevmax,
                                            timetype=timetype)

        obs = ehtim.obsdata.Obsdata(ra, dec, rf, bw, obsarr, self.tarr,
                                    source=str(ra) + ":" + str(dec),
                                    mjd=mjd, timetype=timetype)
        return obs

    def make_subarray(self, sites):
        """Make a subarray from the Array object array that only includes the sites listed in sites.
        """
        all_sites = [t[0] for t in self.tarr]
        mask = np.array([t in sites for t in all_sites])
        return Array(self.tarr[mask])

    def save_txt(self, fname):
        """Save the array data in a text file.
        """
        ehtim.io.save.save_array_txt(self,fname)
        return

###########################################################################################################################################
#Array creation functions
###########################################################################################################################################
def load_txt(fname, ephemdir='ephemeris'):
    """Read an array from a text file and return an Array object.
       Sites with x=y=z=0 are spacecraft, and 2TLE ephemerides are loaded from ephemdir.
    """
    return ehtim.io.load.load_array_txt(fname, ephemdir=ephemdir)
