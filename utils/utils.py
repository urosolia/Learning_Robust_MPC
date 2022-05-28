import numpy as np
import scipy

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