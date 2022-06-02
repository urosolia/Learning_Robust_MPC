import numpy as np
import pdb 
import scipy
from cvxpy import *

class FTOCP(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""
	def __init__(self, N, A, B, Q1, Q, R, x_max, u_max):
		# Define variables
		self.N = N # Horizon Length

		# System Dynamics (x_{k+1} = A x_k + Bu_k)
		self.A = A 
		self.B = B 
		self.n = A.shape[1]
		self.d = B.shape[1]
		self.x_max = x_max
		self.u_max = u_max

		# Cost (h(x,u) = x^TQx +u^TRu)
		self.Q = Q
		self.Q1 = Q1
		self.R = R

		# Initialize Predicted Trajectory
		self.xPred = []
		self.uPred = []

	def solve(self, x0, verbose=False):
		"""This method solves an FTOCP given:
			- x0: initial condition
		""" 

		# Initialize Variables
		x = Variable((self.n, self.N+1))
		u = Variable((self.d, self.N))

		# State Constraints
		constr = [x[:,0] == x0[:]]
		for i in range(0, self.N):
			constr += [x[:,i+1] == self.A@x[:,i] + self.B@u[:,i],
						u[:,i] >= -self.u_max,
						u[:,i] <=  self.u_max,
						x[:,i] >= -self.x_max,
						x[:,i] <=  self.x_max]

		# Cost Function
		cost = 0
		for i in range(0, self.N):
			# Running cost h(x,u) = x^TQx + u^TRu
			cost += quad_form(x[:,i], self.Q) + quad_form(u[:,i], self.R) + norm(self.Q1@x[:,i], 1)
			# cost += norm(self.Q**0.5*x[:,i])**2 + norm(self.R**0.5*u[:,i])**2


		# Solve the Finite Time Optimal Control Problem
		problem = Problem(Minimize(cost), constr)
		problem.solve(verbose=verbose, solver=ECOS)

		if problem.status != "optimal":
			print("problem.status: ", problem.status)

		# Store the open-loop predicted trajectory
		self.xPred = x.value
		self.uPred = u.value	



	

