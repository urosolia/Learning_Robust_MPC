
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools
from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
from cvxpy import *

class LMPC_CT(object):
	"""Learning Model Predictive Controller (LMPC) implemented using the Convex Time varing safe set (CT)
		Inputs:
			- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
			- l: number of past trajectories used to construct the local safe set and local Q-function
			- M: number of data points from each trajectory used to construct the local safe set and local Q-function 
		Methods:
			- addTrajectory: adds a trajectory to the safe set SS and update value function
			- computeCost: computes the cost associated with a feasible trajectory
			- solve: uses ftocp and the stored data to comptute the predicted trajectory
			- closeToSS: computes the K-nearest neighbors to zt"""

	def __init__(self, N, dt, roadHalfWidth, l, P, Xf_vertices = None, verbose = False):
		self.N = N
		self.n = 3
		self.d = 2
		self.radius = 10.0
		self.optCost= np.inf
		self.dt = dt
		self.dimSS = []
		self.roadHalfWidth = roadHalfWidth

		self.SS    = []
		self.uSS   = []
		self.Vfun  = []
		self.l     = l
		self.P     = P
		self.zt    = []
		self.it    = 0
		self.timeVarying = True
		self.itCost = []
		self.verbose = verbose
		self.Xf_vertices = Xf_vertices

	def addTrajectory(self, x, u):
		# Add the feasible trajectory x and the associated input sequence u to the safe set
		self.SS.append(np.array(x).T)
		self.uSS.append(np.concatenate( (np.array(u).T, np.zeros((np.array(u).T.shape[0],1)) ), axis=1)) # Here concatenating zero as f(xf, 0) = xf by assumption

		# Compute and store the cost associated with the feasible trajectory
		self.Vfun.append(np.arange(np.array(x).T.shape[1]-1,-1,-1))

		# Compute initial guess for nonlinear solver and store few variables
		self.xGuess = np.concatenate((np.array(x).T[:,0:(self.N+1)].T.flatten(), np.array(u).T[:,0:(self.N)].T.flatten()), axis = 0)

		# Initialize cost varaibles for bookkeeping
		self.cost    = self.Vfun[-1][0]
		self.itCost.append(self.cost)
		self.optCost = self.cost + 1
		self.oldIt  = self.it
			
		# Pass inital guess to ftopc object
		self.xSol = np.array(x).T[:,0:(self.N+1)]
		self.uSol = np.array(u).T.reshape(self.d,-1)[:,0:(self.N)]

		# Update time Improvement counter
		self.timeImprovement = 0

		# Update iteration counter
		self.it = self.it + 1

		# Update indices of stored data points used to contruct the local safe set and Q-function
		self.SSindices =[]
		Tstar = np.min(self.itCost)
		for i in range(0, self.it):
			Tj = np.shape(self.SS[i])[1]-1
			self.SSindices.append(np.arange(Tj - Tstar + self.N, Tj - Tstar + self.N+self.P))

	def solve(self, xt, verbose = False):		

		# First retive the data points used to cconstruct the safe set.
		minIt = np.max([0, self.it - self.l])
		SSVfun = []
		SSnext = []
		# Loop over j-l iterations used to contruct the local safe set
		for i in range(minIt, self.it):
			# idx associated with the data points from iteration i which are in the local safe set
			idx = self.timeSS(i)
			# Stored state and cost value (Check if using terminal state or terminal point)
			if (self.Xf_vertices is not None) and (idx[-1] == np.shape(self.SS[i])[1]-1):
				augSS   = np.concatenate((self.SS[i][:,idx], self.Xf_vertices), axis=1 )				
				augCost = np.concatenate((self.Vfun[i][idx], np.zeros(self.Xf_vertices.shape[1])), axis=0 )	
				SSVfun.append( np.concatenate( (augSS, [augCost]), axis=0 ).T )

				# Store the successors of the states into the safe set and the control action. (Note that the vertices of X_f are invariant states)
				# This matrix will be used to compute the vector zt which represent a feasible guess for the ftocp at time t+1
				xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
				extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)
				verticesAndInputs = np.concatenate((self.Xf_vertices, np.zeros((self.d, self.Xf_vertices.shape[1]))), axis=0)
				SSnext.append(np.concatenate((extendedSS[:,idx+1], verticesAndInputs), axis = 1).T)
			else:
				SSVfun.append( np.concatenate( (self.SS[i][:,idx], [self.Vfun[i][idx]]), axis=0 ).T )
				# Store the successors of the states into the safe set and the control action. 
				# This matrix will be used to compute a feasible guess for the ftocp at time t+1
				xSSuSS = np.concatenate((self.SS[i], self.uSS[i]), axis = 0)
				extendedSS = np.concatenate((xSSuSS, np.array([xSSuSS[:,-1]]).T), axis=1)
				SSnext.append(extendedSS[:,idx+1].T)


		# From a 3D list to a 2D array
		SSVfun_vector = np.squeeze(list(itertools.chain.from_iterable(SSVfun))).T 
		SSnext_vector = np.squeeze(list(itertools.chain.from_iterable(SSnext))).T 

		# Add dimension if needed
		if SSVfun_vector.ndim == 1:
			SSVfun_vector = np.array([SSVfun_vector]).T

		# Now update ftocp with local safe set
		self.buildNonlinearProgram( SSVfun_vector)

		# Now solve ftocp
		self.solve_ftocp(xt)			
			
		# Assign input
		self.uPred = self.uSol

		# Update guess for the ftocp using optimal predicted trajectory and multipliers lambda 
		if self.optCost > 1:
			xfufNext  = np.dot(SSnext_vector, self.lamb)
			# Update initial guess
			xflatOpenLoop  = np.concatenate( (self.xSol[:,1:(self.N+1)].T.flatten(), xfufNext[0:self.n,0]), axis = 0)
			uflatOpenLoop  = np.concatenate( (self.uSol[:,1:(self.N)].T.flatten()  , xfufNext[self.n:(self.n+self.d),0]), axis = 0)
			self.xGuess = np.concatenate((xflatOpenLoop, uflatOpenLoop) , axis = 0)

	def timeSS(self, it):
		# This function computes the indices used to construct the safe set
		# self.SSindices[it] is initialized when the trajectory is added to the safe set after computing \delta^i and P

		# Read the time indices
		currIdx = self.SSindices[it]
		# By definition we have x_t^j = x_F \forall t > T^j ---> check indices to select
		# currIdxShort = currIdx[ (currIdx >0) & (currIdx < np.shape(self.SS[it])[1])]
		currIdxShort = currIdx[ currIdx < np.shape(self.SS[it])[1] ]

		# Progress time indices
		self.SSindices[it] = self.SSindices[it] + 1

		# If there is just one time index --> add dimension
		if np.shape(currIdxShort)[0] < 1:
			currIdxShort = np.array([np.shape(self.SS[it])[1]-1])

		return currIdxShort


	def set_xf(self, xf):
			# Set terminal state
			if xf.shape[1] >1:
				self.terminalSet = True
				self.xf_lb = xf[:,0]
				self.xf_ub = xf[:,1]
			else:
				self.terminalSet = False
				self.xf    = xf[:,0]
				self.xf_lb = self.xf
				self.xf_ub = self.xf
		
	def checkTaskCompletion(self,x):
		# Check if the task was completed
		taskCompletion = False
		if (self.terminalSet == True) and (self.xf_lb <= x).all() and (x <= self.xf_ub).all():
			taskCompletion = True
		elif (self.terminalSet == False) and np.dot(x-self.xf, x-self.xf)<= 1e-8:
			taskCompletion = True

		return taskCompletion


	def solve_ftocp(self, x0, verbose=False):
		# Initialize initial guess for lambda
		lambGuess = np.hstack( (np.ones(self.dimSS)/self.dimSS, np.zeros(self.n)) )
		lambGuess[0] = 1
		self.xGuessTot = np.hstack( (self.xGuess, lambGuess) )

		# lambGuess = np.concatenate((np.ones(self.dimSS)/self.dimSS, np.zeros(self.n)), axis = 0)
		# lambGuess[0] = 1
		# self.xGuessTot = np.concatenate( (self.xGuess, lambGuess), axis=0 )

		# Need to solve N+1 ftocp as the stage cost is the indicator function --> try all configuration
		costSolved = []
		soluSolved = []
		slackNorm  = []
		for i in range(0, self.N+1): 
			# IMPORTANT: here 'i' represents the number of states constrained to the safe set --> the horizon length is (N-i)
			if i is not self.N:
				# Set box constraints on states (here we constraint the last i steps of the horizon to be xf)
				self.lbx = x0 + [-100, -self.roadHalfWidth, -0]*(self.N-i)+ self.xf_lb.tolist()*i + [-2.0,-1.0]*self.N + [0]*self.dimSS  + [-10]*self.n
				self.ubx = x0 +  [100,  self.roadHalfWidth,  500]*(self.N-i)+ self.xf_ub.tolist()*i + [2.0, 1.0]*self.N + [10]*self.dimSS + [10]*self.n

				# Solve nonlinear programm
				sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics, x0 = self.xGuessTot.tolist())

				# Check if the solution is feasible
				idxSlack = (self.N+1)*self.n + self.d*self.N + self.dimSS
				self.slack    = sol["x"][idxSlack:idxSlack+self.n]
				slackNorm.append(np.linalg.norm(self.slack,2))
				if (self.solver.stats()['success']) and (np.linalg.norm(self.slack,2)< 1e-8):
					self.feasible = 1
					# Notice that the cost is given by the cost of the ftocp + the number of steps not constrainted to be xf
					lamb = sol["x"][((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)]
					terminalCost = np.dot(self.costSS, lamb)
					# costSolved.append(terminalCost+(self.N+1-i))
					if i == 0:
						costSolved.append(terminalCost+(self.N-i))
					else:
						costSolved.append(terminalCost+(self.N-(i-1)))

					soluSolved.append(sol)
				else:
					costSolved.append(np.inf)
					soluSolved.append(sol)
					self.feasible = 0

			else: # if horizon one time step (because N-i = 0) --> just check feasibility of the initial guess
				uGuess = self.xGuess[(self.n*(self.N+1)):(self.n*(self.N+1)+self.d)]
				xNext  = self.f(x0, uGuess)
				slackNorm.append(0.0)
				if self.checkTaskCompletion(xNext):
					self.feasible = 1
					costSolved.append(1)
					sol["x"] = self.xGuessTot
					soluSolved.append(sol)
				else:
					costSolved.append(np.inf)
					soluSolved.append(sol)
					self.feasible = 0


		# Store optimal solution
		self.optCost = np.min(costSolved)
		x = np.array(soluSolved[np.argmin(costSolved)]["x"])
		self.xSol = x[0:(self.N+1)*self.n].reshape((self.N+1,self.n)).T
		self.uSol = x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.N,self.d)).T
		self.lamb = x[((self.N+1)*self.n+self.N*self.d):((self.N+1)*self.n + self.d*self.N + self.dimSS)]
		optSlack = slackNorm[np.argmin(costSolved)]

	def buildNonlinearProgram(self, SSVfun):
		# Define variables
		n = self.n
		d = self.d
		N = self.N
		X      = SX.sym('X', n*(N+1));
		U      = SX.sym('X', d*N);
		dimSS  = np.shape(SSVfun)[1]
		lamb   = SX.sym('X',  dimSS)
		xSS    = SSVfun[0:n, :]
		costSS = np.array([SSVfun[-1, :]])
		slack  = SX.sym('X', n);

		self.dimSS = dimSS
		self.SSVfun = SSVfun
		self.xSS = xSS
		self.costSS = costSS
		self.xSS = xSS
		self.costSS = costSS

		# Define dynamic constraints
		constraint = []
		for i in range(0, N):
			constraint = vertcat(constraint, X[n*(i+1)+0] - (X[n*i+0] + self.dt*X[n*i+2]*np.cos( U[d*i+0] - X[n*i+0] / self.radius) / (1 - X[n*i+1]/self.radius ) )) 
			constraint = vertcat(constraint, X[n*(i+1)+1] - (X[n*i+1] + self.dt*X[n*i+2]*np.sin( U[d*i+0] - X[n*i+0] / self.radius) )) 
			constraint = vertcat(constraint, X[n*(i+1)+2] - (X[n*i+2] + self.dt*U[d*i+1])) 

		# terminal constraints
		constraint = vertcat(constraint, slack + X[n*N:(n*(N+1))+0] - mtimes( xSS ,lamb) )
		constraint = vertcat(constraint, 1 - mtimes(np.ones((1, dimSS )), lamb ) )

		# Defining Cost (We will add stage cost later)
		cost = mtimes(costSS, lamb) + 1000000**2*(slack[0]**2 + slack[1]**2 + slack[2]**2)
		self.constraint= constraint
		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X,U,lamb,slack), 'f':cost, 'g':constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force: 1) n*N state dynamics, 2) n terminal contraints and 3) CVX hull constraint
		self.lbg_dyanmics = [0]*(n*N) + [0]*(n) + [0]
		self.ubg_dyanmics = [0]*(n*N) + [0]*(n) + [0]



	def f(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + self.dt * x[2]*np.cos(u[0] - x[0]/self.radius) / (1 - x[1] / self.radius),
						  x[1] + self.dt * x[2]*np.sin(u[0] - x[0]/self.radius),
						  x[2] + self.dt * u[1]])
		return xNext.tolist()		