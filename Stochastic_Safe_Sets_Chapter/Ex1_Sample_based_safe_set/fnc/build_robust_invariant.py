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

class BuildRobustInvariant(object):

    def __init__(self, barA, barB, A, B, Q, R, bx, bu, verticesW, maxIt, maxTime, maxRollOut, sys_aug):
        # Data roll-outs current iteration
        self.x_data     = [] # list of stored closedLoop
        self.u_data     = [] # list of stored closedLoop

        self.A = A
        self.B = B
        self.barA = barA
        self.barB = barB
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.R = R
        self.maxIt = maxIt
        self.maxTime = maxTime
        self.sys_aug = sys_aug
        self.maxRollOut= maxRollOut

        self.itCounter = 0
        
        self.verticesW = verticesW
        self.computeRobutInvariant()
        self.A_O, self.b_O = pypoman.duality.compute_polytope_halfspaces(self.verticesO)
        self.A_W, self.b_W = pypoman.duality.compute_polytope_halfspaces(self.verticesW)

    def build_robust_invariant(self):
        # Compute robust invariant from data
        x_cl = []; u_cl = [];

        for it in range(0,self.maxIt):
            for rollOut in range(0, self.maxRollOut): # Roll-out loop
                self.sys_aug.reset_IC() # Reset initial conditions
                print("Start roll out: ", rollOut, " of iteration: ", it)
                for t in range(0,self.maxTime): # Time loop
                    # ut = mpc.solve(sys.x[-1])
                    ut = -np.dot(self.K, self.sys_aug.x[-1])
                    self.sys_aug.applyInput(ut)

                # Closed-loop trajectory. The convention is row = time, col = state
                x_cl.append(np.array(self.sys_aug.x))
                self.sys_aug.u.append(-self.K@self.sys_aug.x[-1])
                u_cl.append(np.array(self.sys_aug.u))
                self.add_data(self.sys_aug.x, self.sys_aug.u)
                # TO DO: check if trajectory reached the goal
                # impc.addData(x_cl[-1], u_cl[-1]) # Add data while performing the task
            if self.check_robust_invariance():
                print("Robust invariant found")
                break
            else:
                print("Robust invariant NOT found")
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

    def check_robust_invariance(self):
        robust_invariant = True
        for x in self.x_data:
            for w in self.verticesW:
                x_next = np.dot(self.barAcl,x) + w
                contained = self.check_if_in_cvx_hull(x_next)
                if not contained:
                    robust_invariant = False
                    return robust_invariant

        return robust_invariant

    def computeRobutInvariant(self):
        self.O_v = [np.array([0,0])]
        self.dlqr()
        print("Compute robust invariant")
        # TO DO:
        # - add check for convergence
        # - add check for input and state constraint satifaction
        for i in range(0,20):
            self.O_v = self.MinkowskiSum(self.O_v, self.verticesW.tolist())

        self.verticesO = np.array(self.O_v)

    def MinkowskiSum(self, setA, setB):
        vertices = []
        for v1 in setA:
            for v2 in setB:
                vertices.append(np.dot(self.Acl,v1) + v2)

        cvxHull = ConvexHull(vertices)
        verticesOut = []
        for idx in cvxHull.vertices:
            verticesOut.append(vertices[idx])

        return verticesOut

    def dlqr(self):
        # solve the ricatti equation
        P = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        # compute the LQR gain
        self.K   = np.array(scipy.linalg.inv(self.B.T*P*self.B+self.R)*(self.B.T*P*self.A))
        self.Acl = self.A - np.dot(self.B, self.K)
        self.barAcl = self.barA - np.dot(self.barB, self.K)

    def shrink_constraint(self):
        array_data = np.array(self.x_data).T
        self.bx_shrink = []
        self.bx_shrink.append(self.bx[0]-array_data.max(axis=1))
        self.bx_shrink.append(self.bx[1]-array_data.min(axis=1))

        print("shrunk state constraints max: ", self.bx_shrink[0])
        print("shrunk state constraints min: ", self.bx_shrink[1])
        
        array_data_kx = np.dot(self.K, array_data)

        self.bu_shrink = []
        self.bu_shrink.append(self.bu[0]-array_data_kx.max(axis=1))
        self.bu_shrink.append(self.bu[1]-array_data_kx.min(axis=1))

        print("shrunk input constraints max: ", self.bu_shrink[0])
        print("shrunk input constraints min: ", self.bu_shrink[1])
