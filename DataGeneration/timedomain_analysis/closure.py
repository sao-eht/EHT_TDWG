from __future__ import division
from __future__ import print_function

import numpy as np
import itertools as it

class Closure(object):

    def __init__(self, obs):
        sites = [obs.tarr[i][0] for i in range(len(obs.tarr))] # List of telescope names
        tri  = list(it.combinations(sites,3)) # List of all possible triangles
        quad = list(it.combinations(sites,4)) # List of all possible quadrangles

        # closure phase/amp. time curves of all possible tri/quadr-angles ("None" if no data)
        cp = obs.get_cphase_curves(tri)
        ca = obs.get_camp_curves(quad)

        # remove tri/quadr-angles that have no data
        self.cp=[]
        self.tri=[]
        for i in range(len(cp)):
            if cp[i] is not None:
                (self.cp).append(cp[i])
                (self.tri).append(tri[i])
        self.ca=[]
        self.quad=[]
        for i in range(len(ca)):
            if ca[i] is not None:
                (self.ca).append(ca[i])
                (self.quad).append(quad[i])


    def record_cp( self, tri_id ):
        cp = np.array(self.cp[tri_id])
        if cp is None:
            print("cp[%d] does not have observational data"%tri_id)
            return

        fname = "cp_%s-%s-%s"%(self.tri[tri_id][0],self.tri[tri_id][1],self.tri[tri_id][2])
        f = open(fname,"w")
        for i in range(len(cp[0])):
            f.write("%f %f %f\n"%(cp[0][i],cp[1][i],cp[2][i]))
        f.close()


    def record_ca( self, quad_id ):
        ca = np.array(self.ca[quad_id])
        if ca is None:
            print("ca[%d] does not have observational data"%quad_id)
            return

        fname = "ca_%s-%s-%s-%s"%(self.quad[quad_id][0],self.quad[quad_id][1],self.quad[quad_id][2],self.quad[quad_id][3])
        f = open(fname,"w")
        for i in range(len(ca[0])):
            f.write("%f %f %f\n"%(ca[0][i],ca[1][i],ca[2][i]))
        f.close()

