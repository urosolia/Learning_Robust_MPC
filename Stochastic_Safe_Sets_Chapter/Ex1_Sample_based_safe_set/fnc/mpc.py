from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *
import time

class MPC(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0 and terminal contraints
		- buildNonlinearProgram: builds the ftocp program solved by the above solve method
		- model: given x_t and u_t computes x_{t+1} = f( x_t, u_t )

	"""

	def __init__(self, N, A, B, Q, R, Qf, bx, bu, K, verticesO, verticesXf):
		# Define variables
		self.A = A
		self.B = B
		self.N  = N
		self.n  = A.shape[1]
		self.d  = B.shape[1]
		self.bx = bx
		self.bu = bu
		self.Q = Q
		self.Qf = Qf
		self.R = R
		self.K = K
		self.verticesO = verticesO
		self.verticesXf = verticesXf

		print("Initializing FTOCP")
		self.buildFTOCP()
		self.solverTime = []
		print("Done initializing FTOCP")

	def solve(self, x0, verbose=False):
			# Set initial condition + state and input box constraints
			self.lbx = x0.tolist() + (self.bx[1]).tolist()*(self.N+1) + (self.bu[1]).tolist()*self.N + [0]*self.verticesO.shape[0] + [0]*self.verticesXf.shape[0]
			self.ubx = x0.tolist() + (self.bx[0]).tolist()*(self.N+1) + (self.bu[0]).tolist()*self.N + [1]*self.verticesO.shape[0] + [1]*self.verticesXf.shape[0]

			# Solve nonlinear programm
			start = time.time()
			sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
			end = time.time()
			self.solverTime = end - start
			# print("solver time: ", self.solverTime)

			# Check if the solution is feasible
			if (self.solver.stats()['success']):
				self.feasible = 1
				x = sol["x"]
				self.xt    = np.array(x[0:self.n])
				self.xPred = np.array(x[self.n:(self.N+2)*self.n].reshape((self.n,self.N+1))).T
				self.uPred = np.array(x[(self.N+2)*self.n:((self.N+2)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
				self.lambda_xt = np.array(x[((self.N+2)*self.n + self.d*self.N):((self.N+2)*self.n + self.d*self.N)+self.verticesO.shape[0]])
				self.lambda_xf = np.array(x[((self.N+2)*self.n + self.d*self.N)+self.verticesO.shape[0]:((self.N+2)*self.n + self.d*self.N)+self.verticesO.shape[0]+self.verticesXf.shape[0]])
				
				self.mpcInput = self.uPred[0][0] - self.K @ (self.xt.squeeze() - self.xPred[0,:])
			else:
				self.xPred = np.zeros((self.N+1,self.n) )
				self.uPred = np.zeros((self.N,self.d))
				self.mpcInput = []
				self.feasible = 0
				print("Unfeasible")
				
			return self.mpcInput

	def buildFTOCP(self):
		# Define variables
		n  = self.n
		d  = self.d

		# Define variables
		Xt      = SX.sym('Xt', n)
		X      = SX.sym('X', n*(self.N+1))
		U      = SX.sym('U', d*self.N)
		lamb_xt   = SX.sym('lamb_xt', self.verticesO.shape[0])
		lamb_xf   = SX.sym('lamb_xt', self.verticesXf.shape[0])
		
		# Define dynamic constraints
		self.constraint = []
		for i in range(0, self.N):
			X_next = self.dynamics(X[n*i:n*(i+1)], U[d*i:d*(i+1)])
			for j in range(0, self.n):
				self.constraint = vertcat(self.constraint, X_next[j] - X[n*(i+1)+j] ) 

		# Initial constraint x{t|t} - x(t) \in \mathcal{O}
		self.constraint = vertcat(self.constraint, Xt - X[0:n] - mtimes( self.verticesO.T ,lamb_xt) )
		self.constraint = vertcat(self.constraint, 1 - mtimes(np.ones((1, self.verticesO.shape[0] )), lamb_xt ) )

		# Enforce terminal constraint
		self.constraint = vertcat(self.constraint, X[n*self.N:(n*(self.N+1))] - mtimes( self.verticesXf.T ,lamb_xf) )
		self.constraint = vertcat(self.constraint, 1 - mtimes(np.ones((1, self.verticesXf.shape[0])), lamb_xf ) )

		# Defining Cost (We will add terminal cost later)
		# self.cost = (self.K @ (Xt - X[0:n])).T @ self.R @ (self.K @ (Xt - X[0:n]))
		# self.cost = self.cost + (Xt - X[0:n]).T @ self.Q @ (Xt - X[0:n])
		self.cost = 0
		for i in range(0, self.N):
			self.cost = self.cost + X[n*i:n*(i+1)].T @ self.Q @ X[n*i:n*(i+1)]
			self.cost = self.cost + U[d*i:d*(i+1)].T @ self.R @ U[d*i:d*(i+1)]

		self.cost = self.cost + X[n*self.N:n*(self.N+1)].T @ self.Qf @ X[n*self.N:n*(self.N+1)]

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#\\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(Xt, X, U, lamb_xt, lamb_xf), 'f':self.cost, 'g':self.constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force n*N state dynamics, n+1 for initial constraint, and [0]*(n+1) for terminal constraint
		self.lbg_dyanmics = [0]*(n*self.N) + [0]*(n+1) + [0]*(n+1)
		self.ubg_dyanmics = [0]*(n*self.N) + [0]*(n+1) + [0]*(n+1)

	def dynamics(self, x, u):
		return self.A @ x + self.B @ u
