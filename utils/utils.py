import numpy as np
import scipy

class system(object):
	"""docstring for system"""
	def __init__(self, A, B, w_inf, x0):
		self.A     = A
		self.B     = B
		self.w_inf = w_inf
		self.x 	   = [x0]
		self.u 	   = []
		self.w 	   = []
		self.x0    = x0

		self.w_v = w_inf*(2*((np.arange(2**A.shape[1])[:,None] & (1 << np.arange(A.shape[1]))) > 0) - 1)
		print("Disturbance vertices \n", self.w_v )
		
	def applyInput(self, ut):
		self.u.append(ut)
		self.w.append(np.random.uniform(-self.w_inf,self.w_inf,self.A.shape[1]))
		xnext = np.dot(self.A,self.x[-1]) + np.dot(self.B,self.u[-1]) + self.w[-1]
		self.x.append(xnext)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []
		
def dlqr(A, B, Q, R, verbose = False):
	# solve the ricatti equation
	P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	# compute the LQR gain
	K   = np.array(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
	Acl = A - np.dot(B, K)

	if verbose == True:
		print("P: ", P)
		print("K: ", K)
		print("Acl: ", Acl)
	return P, K, Acl