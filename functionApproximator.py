#!/usr/local/bin/python
"""
-----Put your information here-----
	
	authors: "???", "???", "???"
	emails: "???", "???", "???"

-----------------------------------
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

class fa():
	
	def __init__(self, numberOfFeatures, learningRate=0.01, minDelta=0.00001, maxIteration = 20000):
		"""
		Initialize class member data based on user defined variables.
		
		:param numberOfFeatures: Number of features/basis functions used in the FA.
		:param learningRate: Indicated as "alpha" in the gradient descent method. Determines the "distance" traveled in the direction of the gradient for each step. (Only used in Gradient Descent. Defaults to 0.01)
		:param minDelta: Minimum change between thetas between two consecutive iterations. This is used as a convergence criterion. (Iterative methods only. Defaults to 0.00001)
		:param maxIteration: Maximum number of iterations used for training the FA. (Iterative methods only. Defaults to 20000)
		
		:returns: A function approximator object with the desired variable initializations.
		"""
		
		# Variables pertinent to the core learning algorithms
		self.numFeatures 	= numberOfFeatures 
		self.theta 			= np.zeros(self.numFeatures,) #initialize theta vector to appropriate size
		self.maxIter 		= maxIteration
		self.threshold 		= minDelta
		self.alpha 			= learningRate
		
		# Auxiliary variables used in the class
		self.theta_old 		= np.zeros((self.numFeatures,))
		self.delta 			= 100
		self.iterationCount = 0
		self.deltaHistory 	= []
		self.cpuHistory 	= []
		self.memHistory 	= []
		self.timeHistory 	= []
		self.trainingTime 	= 0.0
		self.method 		= 'None'
		self.thetaHistory 	= []
		self.thetaHistory.append(self.theta)
		self.performance()
		# Set Centers and Widths
		self.setCentersAndWidths()
		
		
		
		
##########################################################################################		
############################### Various Training Methods #################################
##########################################################################################	
	
	######################
	## Gradient Descent ##
	######################

	def train_GD(self):
		self.method = 'Incremental'		
		
		# Initialize theta_0 
		self.theta = """???"""
		
		while self.delta >= self.threshold and self.iterationCount < self.maxIter:
			# Draw a random sample on the interval [0,1]
			x = np.random.random() 
			y = self.generateDataSample(x)
				
			self.theta_old = self.theta
				
			t0 = time.time()
			#----------------------#
			## Training Algorithm ##
			#----------------------#
				
			self.theta = self.theta_old - """???"""
			
			#-----------------------------#
			## End of Training Algorithm ##
			#-----------------------------#
				
			t1 = time.time()
			self.timeHistory.append(t1 - t0)
			self.performance()
				
			self.thetaHistory.append(self.theta)
			self.delta = self.calculateDelta()
			self.deltaHistory.append(self.delta)
			self.iterationCount+=1
			self.printStats()
		

		
	###################
	## Least Squares ##
	###################
		
	def train_LS(self, data):
				
		#Get x and y values separated from 'data'
		xData = data[0,:]
		yData = data[1,:]
		
		t0 = time.time()
		
		#----------------------#
		## Training Algorithm ##
		#----------------------#
				
		self.theta = """???"""
		
		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#
		
		t1 = time.time()
		self.timeHistory.append(t1 - t0)
		self.performance()
		
	#############################
	## Recursive Least Squares ##
	#############################
		
	def train_RLS(self):
		self.method = 'Incremental'	
				
		## Initialize A and b ##
		A = """???"""
		b = """???"""
		
		# Begin training
		while self.delta >= self.threshold and self.iterationCount < self.maxIter:
		
			# Draw a random sample on the interval [0,1]
			x = np.random.random() 
			y = self.generateDataSample(x)
			
			t0 = time.time()
			
			#----------------------#
			## Training Algorithm ##
			#----------------------#
				
			self.theta = """???"""
			
			#-----------------------------#
			## End of Training Algorithm ##
			#-----------------------------#
				
			t1 = time.time()
			self.timeHistory.append(t1 - t0)
			self.performance()
				
			self.thetaHistory.append(self.theta)
			self.delta = self.calculateDelta()
			self.theta_old = self.theta
			self.deltaHistory.append(self.delta)
			self.iterationCount+=1
			self.printStats()
	
	################################################################
	## Recursive Least Squares Version 2 (Ainv estimation method) ##
	################################################################
	
	def train_RLS2(self):
		self.method = 'Incremental'		
				
		## Initialize Ainv and b ##
		Ainv = """???"""
		b = """???"""
		
		# Begin training
		while self.delta >= self.threshold and self.iterationCount < self.maxIter:
		
			# Draw a random sample on the interval [0,1]
			x = np.random.random() 
			y = self.generateDataSample(x)
			
			t0 = time.time()
			
			#----------------------#
			## Training Algorithm ##
			#----------------------#
				
			self.theta = """???"""
				
			#-----------------------------#
			## End of Training Algorithm ##
			#-----------------------------#
				
			t1 = time.time()
			self.timeHistory.append(t1 - t0)
			self.performance()
				
			self.thetaHistory.append(self.theta)
			self.delta = self.calculateDelta()
			self.theta_old = self.theta
			self.deltaHistory.append(self.delta)
			self.iterationCount+=1
			self.printStats()


##########################################################################################		
############################ End of Training Method Code #################################
##########################################################################################
 
 	def generateDataSample(self, x):
 		"""
		Generate a noisy data sample from a given data point in the range [0,1]
		
		:param x: A scalar dependent variable for which to calculate the output y_noisy
		
		:returns: The output of the function f with gaussian noise added
		
		"""
 		y = 1 - x - math.sin(-2*math.pi*x**3)*math.cos(-2*math.pi*x**3)*math.exp(-x**4)
		sigma = 0.1
		noise = sigma * np.random.random()
		y_noisy = y + noise
		return y_noisy
		
	def setCentersAndWidths(self):
		"""
		Set the center location and width for each basis function assuming the dependent variable
		range is [0,1].
		"""
		xMin = 0.0
		xMax = 1.0
		self.centers = np.linspace(xMin, xMax, self.numFeatures)
		self.widthConstant = (xMax - xMin) / self.numFeatures / 10
		self.widths = np.ones(self.numFeatures,) * self.widthConstant
 
	
	def featureOutput(self, input):
		"""
		Get the output of the features for a given input variable(s)
		
		:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
		
		:returns: A vector of feature outputs with size [NumFeats x Ns]
		"""
		if np.size(input) == 1: 
			phi = np.exp(-np.divide(np.square(input - self.centers), self.widths))
		
		elif np.size(input) > 1:
			numEvals = np.shape(input)[0]
			#Repeat vectors to vectorize output calculation
			inputMat = np.array([input,]*self.numFeatures)
			centersMat = np.array([self.centers,]*numEvals).transpose() 
			widthsMat = np.array([self.widths,]*numEvals).transpose() 
			phi = np.exp(-np.divide(np.square(inputMat - centersMat), widthsMat))
			
		return phi
	
	
	def functionApproximatorOutput(self, input, *user_theta):
		"""
		Get the FA output for a given input variable(s)
		
		:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
		:param user_theta: (Variable argument) A vector of theta variables to apply to the FA. If left blank the method will default to using the trained thetas in self.theta. This is only used for visualization purposes.
		
		:returns: A vector of function approximator outputs with size [Ns]
		"""
		phi = self.featureOutput(input)
		
		if not user_theta:
			Theta = self.theta
		else:
			Theta = np.array(user_theta)
		
		if np.size(input) == 1:
			fa_out = np.dot(phi, Theta.transpose())
		elif np.size(input) > 1:
			fa_out = np.dot(phi.transpose(), Theta.transpose()) 
		
		return fa_out
		
	
	def calculateDelta(self):
		"""
		Function used to calculate the change in characteristic variables between iterations. 
		Used to estimate the convergence of the iterative learning methods.
		
		:returns: A scalar estimation of the difference between consecutive iterations
		"""
		#delta = math.fabs(np.linalg.norm(self.theta - self.theta_old))
		delta = np.mean(np.abs(self.theta - self.theta_old))
		
		#xData = data[0,:]
		#yData = data[1,:]
		#delta = np.linalg.norm(yData - self.functionApproximatorOutput(xData))
		
		return delta
		
	
	def printStats(self):
		"""
		Print various iteration/convergence statistics during training
		"""
		print 'Iteration: ', self.iterationCount, ' Delta: ', self.delta
		
		
	def performance(self):
		"""
		Get Cpu and RAM usage statistics using the psutil library
		"""
		if psutil.__version__ == '1.1.2':
			self.memHistory.append(psutil.virtual_memory().percent)
			self.cpuHistory.append(psutil.cpu_percent(interval=None, percpu=True))
		else:
			self.memHistory.append([0])
			self.cpuHistory.append([0])
		
			
	
		
