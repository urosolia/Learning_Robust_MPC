import numpy as np

class SIMULATOR(object):
	""" Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""
	def __init__(self, system, A = [], B = [], radius = [], dt = 0.1,  map = []):
		# Define variables
		self.system = system
		self.A = A
		self.B = B
		self.radius = radius
		self.dt = dt
		self.map = map

	def sim(self, x, u):
		if self.system == "linear_system":
			x_next = self.linear_system(x,u)
		elif self.system == "unicycle":
			x_next = self.unicycle(x,u)
		elif self.system == "dyn_bicycle_model":
			x, x_glob = self.dyn_bicycle_model(x, u)
			x_next = [x, x_glob]
		return x_next

	def linear_system(self, x,u):
		return (np.dot(self.A,x) + np.squeeze(np.dot(self.B,u))).tolist()

	def unicycle(self, x, u):
		# Given a state x and input u it return the successor state
		xNext = np.array([x[0] + self.dt * x[2]*np.cos(u[0] - x[0]/self.radius) / (1 - x[1] / self.radius),
						  x[1] + self.dt * x[2]*np.sin(u[0] - x[0]/self.radius),
						  x[2] + self.dt * u[1]])
		return xNext.tolist()


	# Introduce function for computing road edges
	def computeRoadEdges(self, s_start, s_end, circleRadius, roadHalfWidth, signEdge = 1, disc = 1):
		edges = []
		for k in np.arange(s_start, s_end+disc, disc):#in range(s_start*disc, s_end*disc):
			angle  = k/circleRadius
			radius = circleRadius  + signEdge * roadHalfWidth
			edges.append([radius*np.sin(angle), circleRadius-radius*np.cos(angle)])

		return np.array(edges)

	# Introduce function for change of coordinates from curvilinear absicssa to XY
	def from_curvilinear_to_xy(self, xcl_feasible):
		feasibleTraj = []
		for k in range(0, np.shape(np.array(xcl_feasible))[0]):
			angle  = np.array(xcl_feasible)[k, 0]/self.radius
			radius_curr = self.radius  - np.array(xcl_feasible)[k, 1]
			feasibleTraj.append([radius_curr*np.sin(angle), self.radius-radius_curr*np.cos(angle)])

		return feasibleTraj


	def dyn_bicycle_model(self, x_states_list, u):
		# This method computes the system evolution. Note that the discretization is deltaT and therefore is needed that
		# dt <= deltaT and ( dt / deltaT) = integer value
		x = np.array(x_states_list[0])
		x_glob = np.array(x_states_list[1])

		# Vehicle Parameters
		m  = 1.98
		lf = 0.125
		lr = 0.125
		Iz = 0.024
		Df = 0.8 * m * 9.81 / 2.0
		Cf = 1.25
		Bf = 1.0
		Dr = 0.8 * m * 9.81 / 2.0
		Cr = 1.25
		Br = 1.0

		# Discretization Parameters
		deltaT = 0.001
		x_next	 = np.zeros(x.shape[0])
		cur_x_next = np.zeros(x.shape[0])

		# Extract the value of the states
		delta = u[0]
		a	 = u[1]

		psi = x_glob[3]
		X = x_glob[4]
		Y = x_glob[5]

		vx	= x[0]
		vy	= x[1]
		wz	= x[2]
		epsi  = x[3]
		s	 = x[4]
		ey	= x[5]

		# Initialize counter
		i = 0
		while( (i+1) * deltaT <= self.dt):
			# Compute tire split angle
			alpha_f = delta - np.arctan2( vy + lf * wz, vx )
			alpha_r = - np.arctan2( vy - lf * wz , vx)

			# Compute lateral force at front and rear tire
			Fyf = Df * np.sin( Cf * np.arctan(Bf * alpha_f ) )
			Fyr = Dr * np.sin( Cr * np.arctan(Br * alpha_r ) )

			# Propagate the dynamics of deltaT
			x_next[0] = vx  + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
			x_next[1] = vy  + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
			x_next[2] = wz  + deltaT * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
			x_next[3] = psi + deltaT * (wz)
			x_next[4] =   X + deltaT * ((vx * np.cos(psi) - vy * np.sin(psi)))
			x_next[5] =   Y + deltaT * (vx * np.sin(psi)  + vy * np.cos(psi))

			cur = self.map.curvature(s)
			cur_x_next[0] = vx   + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
			cur_x_next[1] = vy   + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
			cur_x_next[2] = wz   + deltaT * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
			cur_x_next[3] = epsi + deltaT * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
			cur_x_next[4] = s	+ deltaT * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
			cur_x_next[5] = ey   + deltaT * (vx * np.sin(epsi) + vy * np.cos(epsi))

			# Update the value of the states
			psi  = x_next[3]
			X	= x_next[4]
			Y	= x_next[5]

			vx   = cur_x_next[0]
			vy   = cur_x_next[1]
			wz   = cur_x_next[2]
			epsi = cur_x_next[3]
			s	= cur_x_next[4]
			ey   = cur_x_next[5]

			# Increment counter
			i = i+1

		# Noises
		noise_vx = np.max([-0.05, np.min([np.random.randn() * 0.01, 0.05])])
		noise_vy = np.max([-0.05, np.min([np.random.randn() * 0.01, 0.05])])
		noise_wz = np.max([-0.05, np.min([np.random.randn() * 0.005, 0.05])])

		cur_x_next[0] = cur_x_next[0] + 0.01*noise_vx
		cur_x_next[1] = cur_x_next[1] + 0.01*noise_vy
		cur_x_next[2] = cur_x_next[2] + 0.01*noise_wz

		x_next[0] = x_next[0] + 0.01*noise_vx
		x_next[1] = x_next[1] + 0.01*noise_vy
		x_next[2] = x_next[2] + 0.01*noise_wz

		return cur_x_next.tolist(), x_next.tolist()

	
   