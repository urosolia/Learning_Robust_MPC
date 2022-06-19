
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools
import numpy as np
import pdb 
import scipy
from cvxpy import *

class LMPC_CS(object):
	"""Learning Model Predictive Controller (LMPC) implemented using the Convex Safe set (CS)
	Inputs:
		- A,B: System dyamics
		- x_max, u_max: max of the infinity norm constraint
		- Q1, Q, R: cost matrices
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""
	def __init__(self, N, A, B, Q1, Q, R, x_max, u_max, verbose = False):
		# Initialization
		self.N = N # Horizon Length
		self.it_cost = []
		self.Q1 = Q1
		self.Q = Q
		self.R = R
		self.A = A 
		self.B = B
		self.x_max = x_max
		self.u_max = u_max
		self.n = A.shape[1]
		self.d = B.shape[1]
		self.SS    = np.zeros((self.n,1))
		self.uSS   = np.zeros((self.d,1))
		self.Vfun  = np.zeros(1)
		self.verbose = False

	def addTrajectory(self, x, u):
		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS = np.hstack((self.SS, np.array(x).T))
		self.uSS = np.hstack((self.uSS, np.array(u).T.reshape(self.d,-1)))

		# Compute and store the cost associated with the feasible trajectory
		cost = self.computeCost(x, u)
		self.Vfun = np.append(self.Vfun, np.array(cost))

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
		self.it_cost.append(cost[0])
		if self.verbose == True:
			print("Trajectory added to the Safe Set. Current Iteration: ", len(self.it_cost))
			print("Performance stored trajectories: \n", [cost for cost in self.it_cost])

	def computeCost(self, x, u):
		# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
		for i in range(0,len(x)):
			idx = len(x)-1 - i
			if i == 0:
				cost = [np.dot(np.dot(x[idx],self.Q),x[idx]) + np.linalg.norm(self.Q1@x[idx], 1)]
			else:
				cost.append(float(np.dot(np.dot(x[idx],self.Q),x[idx]) + np.linalg.norm(self.Q1@x[idx], 1) + np.dot(np.dot(u[idx],self.R),u[idx]) + cost[-1]))
		
		# Finally flip the cost to have correct order
		return np.flip(cost).tolist()

	def solve(self, x0, verbose = False):
		"""This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Vfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Vfun)
		""" 

		# Initialize Variables
		x = Variable((self.n, self.N+1))
		u = Variable((self.d, self.N))
		lambVar = Variable((self.SS.shape[1], 1)) # Initialize vector of variables

		# State Constraints
		constr = [x[:,0] == x0[:]]
		for i in range(0, self.N):
			constr += [x[:,i+1] == self.A @ x[:,i] + self.B @ u[:,i],
						u[:,i] >= -self.u_max,
						u[:,i] <=  self.u_max,
						x[:,i] >= -self.x_max,
						x[:,i] <=  self.x_max]

		# Enforce terminal state into the convex safe set
		constr += [self.SS @ lambVar[:,0] == x[:,self.N], # Terminal state \in ConvHull(SS)
					np.ones((1, self.SS.shape[1])) @ lambVar[:,0] == 1, # Multiplies \lambda sum to 1
					lambVar >= 0] # Multiplier are positive definite

		# Cost Function
		cost = 0
		for i in range(0, self.N):
			# Running cost h(x,u) = x^TQx + u^TRu
			cost += quad_form(x[:,i], self.Q) + norm(self.Q1@x[:,i], 1) + quad_form(u[:,i], self.R)
			
		# Terminal cost if SS not empty
		cost += self.Vfun @ lambVar[:,0]  # It terminal cost is given by interpolation using \lambda		

		# Solve the Finite Time Optimal Control Problem
		problem = Problem(Minimize(cost), constr)
		problem.solve(verbose=verbose, solver=ECOS)
		if problem.status != "optimal":
			print("problem.status: ", problem.status)

		# Store the open-loop predicted trajectory
		self.xPred = x.value
		self.uPred = u.value	
		# self.lamb  = lambVar.value
