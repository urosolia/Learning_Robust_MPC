class MIX_OL_CL(object):
	def __init__(self, radius):
		self.radius = radius

	def solve(self, x, t, totTimeSteps):
		self.uPred = [0, 0]
		if t ==0:
			self.uPred[1] =  0.25
		elif t== 1:
			self.uPred[1] =   0.25
		elif t== 2:
			self.uPred[1] =   0.25
		elif t==(totTimeSteps-5):
			self.uPred[1] =  -0.25
		elif t==(totTimeSteps-4):
			self.uPred[1] =  -0.25
		elif t==(totTimeSteps-3):
			self.uPred[1] =  -0.25
		else:
			self.uPred[1] = 0   
		
		self.uPred[0] =  x[0] / self.radius;

