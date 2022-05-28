import pypoman
from numpy import array, eye, ones, vstack, zeros
import numpy as np
import scipy
import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d

class polytope(object):
	"""docstring for polytope"""
	def __init__(self, F=None, b=None, vertices=None):

		if vertices is None:
			self.b = b
			self.F = F
			self.computeVRep()
		else:
			hull= ConvexHull(vertices)
			self.vertices = vertices[hull.vertices, :]
			self.vertices = np.round(self.vertices, 6)

			try:
				self.computeHRep()
				self.computeVRep()
			except:
				print("Numerical inaccuracy while computing HRep and/or VRep. Rounding vertices to 6th significant digit.")
				self.vertices = np.round(self.vertices, 6)
				self.computeHRep()
				self.computeVRep()


	def NStepPreAB(self, A, B, Fx, bx, Fu, bu, N):

		for i in range(0, N):
			FPreAB, bPreAB = self.preAB(A, B, Fu, bu)
			FPreAB = np.vstack((FPreAB, Fx))
			bPreAB = np.hstack((bPreAB, bx))
			self.F = FPreAB
			self.b = bPreAB

	def computeO_inf(self, A):
		for i in range(0,5):
			Fpre, bpre = self.preA(A)
			self.intersect(Fpre, bpre)
	
	def computeC_inf(self, A, B):
		for i in range(0,5):
			Fpre, bpre = self.preAB(A, B, self.F.shape[1])
			self.intersect(Fpre, bpre)

	def preA(self, A):
		b = self.b
		F = np.dot(self.F, A)	
		return F, b
	
	def intersect(self, F_intersect, b_intersect):
		self.F = np.vstack((self.F, F_intersect))
		self.b = np.hstack((self.b, b_intersect))
		
	def computeVRep(self, verbose = False):
		self.vertices = pypoman.duality.compute_polytope_vertices(self.F, self.b)
		if verbose == True: print("vertices: ", self.vertices)

	def computeHRep(self):
		self.F, self.b = pypoman.duality.compute_polytope_halfspaces(self.vertices)

	def preAB(self, A, B, Fu, bu):
		n = A.shape[1] 
		d = B.shape[1]

		# Original polytope:
		F1 = np.hstack( ( np.dot(self.F, A), np.dot(self.F, B) ) )
		b1 = self.b
		
		F2 = np.hstack( ( np.zeros((Fu.shape[0], n)), Fu ) )
		b2 = bu
		ineq = (np.vstack((F1, F2)), np.hstack(( b1, b2 )) )  # A * x + Bu <= b, F_u u <= bu 

		# Projection is proj(x) = [x_0 x_1]
		E          = np.zeros(( n, n+d ))
		E[0:n,0:n] = np.eye(n)
		f          = np.zeros(n)
		proj       = (E, f)  # proj(x) = E * x + f

		vertices = pypoman.project_polytope(proj, ineq)#, eq=None, method='bretl')
		F, b = pypoman.duality.compute_polytope_halfspaces(vertices)
		return F, b

	def plot2DPolytope(self, color, label = None):
		# This works only in 2D!!!!
		vertices  = pypoman.polygon.compute_polygon_hull(self.F, self.b)
		vertices.append(vertices[0])
		xs, ys = zip(*vertices) #create lists of x and y values
		plt.plot(xs, ys, '-o', color=color, label=label)