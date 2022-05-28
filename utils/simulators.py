import numpy as np

class SIMULATOR(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""
	def __init__(self, system, A = [], B = []):
		# Define variables
		self.system = system
		self.A = A
		self.B = B

	def sim(self, x, u):
		if self.system == "linear_system":
			x_next = self.linear_system(x,u)
		return x_next

	def linear_system(self, x,u):
		return (np.dot(self.A,x) + np.squeeze(np.dot(self.B,u))).tolist()