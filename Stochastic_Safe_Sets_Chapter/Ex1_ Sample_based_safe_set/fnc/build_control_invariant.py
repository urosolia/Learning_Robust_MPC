from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers
import pypoman
import scipy
solvers.options['show_progress'] = False
from scipy.optimize import linprog

class BuildControlInvariant(object):

    def __init__(self, A, B, maxIt, maxTime, maxRollOut, sys, mpc, x_data_O_estimated, u_data_O_estimated, store_all_data_flag=False):
        # Data roll-outs current iteration
        self.x_data     = x_data_O_estimated # list of stored closedLoop
        self.u_data     = u_data_O_estimated # list of stored closedLoop

        self.x_cl_data = []
        self.u_cl_data = []

        self.store_all_data_flag = store_all_data_flag

        self.A = A
        self.B = B

        self.mpc = mpc

        self.maxIt = maxIt
        self.maxTime = maxTime
        self.sys = sys
        self.maxRollOut= maxRollOut

        
    def build_control_invariant(self):
        # Compute robust invariant from data
        x_cl = []; u_cl = [];

        for it in range(0,self.maxIt):
            for rollOut in range(0, self.maxRollOut): # Roll-out loop
                self.sys.reset_IC() # Reset initial conditions
                print("Start roll out: ", rollOut, " of iteration: ", it)
                for t in range(0,self.maxTime): # Time loop
                    ut = self.mpc.solve(self.sys.x[-1])
                    self.sys.applyInput(ut)

                # Closed-loop trajectory. The convention is row = time, col = state
                x_cl.append(np.array(self.sys.x))
                u_cl.append(np.array(self.sys.u))
                self.add_data(self.sys.x, self.sys.u)

                if self.store_all_data_flag == True:
                    self.x_cl_data.append(np.array(self.sys.x))
                    self.u_cl_data.append(np.array(self.sys.u))


            if self.check_control_invariance():
                print("Control invariant found")
                break
            else:
                print("Control invariant NOT found")
        self.x_cl = x_cl
        self.u_cl = u_cl

    def add_data(self, x_cl, u_cl):
        for x, u in zip(x_cl, u_cl):
            if len(self.x_data)>0:
                contained = self.check_if_in_cvx_hull(x)
                if contained == False:
                    self.x_data.append(x)
                    self.u_data.append(u)
            else:
                self.x_data.append(x)
                self.u_data.append(u)

    def check_if_in_cvx_hull(self, x):
        points = np.array(self.x_data)
        n_points = len(points)
        c = np.zeros(n_points)
        A = np.r_[points.T,np.ones((1,n_points))]
        b = np.r_[x, np.ones(1)]
        lp = linprog(c, A_eq=A, b_eq=b)
        return lp.success

    def check_control_invariance(self):
        control_invariant = True
        for x, u in zip(self.x_data, self.u_data):
            x_next = np.dot(self.A, x) + np.dot(self.B, u)
            contained = self.check_if_in_cvx_hull(x_next)
            if not contained:
                control_invariant = False
                return control_invariant

        return control_invariant