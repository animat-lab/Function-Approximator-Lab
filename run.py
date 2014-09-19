#!/usr/local/bin/python
"""
-----Put your information here-----
	
	authors: "???", "???", "???"
	emails: "???", "???", "???"

-----------------------------------
"""

from functionApproximator import *
from functionApproximator_PlottingTools import *
from functionApproximator_LW import *



### Load data set to be approximated (for batch methods)
############################################################################


filename = './noisyDataSet.txt'
data = np.loadtxt(filename)


### Set up FA variables
############################################################################

# Number of features/basis functions used in the FA.
numFeats = 10 

# Indicated as "alpha" in the gradient descent method. Determines the "distance" traveled in the direction of the gradient for each step.
# (Only used in Gradient Descent)
# (Defaults to 0.01)
learningRate = 0.01

# Minimum change between thetas between two consecutive iterations. This is used as a convergence criterion.
# (Iterative methods only)
# (Defaults to 0.00001)
minDelta = 1E-12

# Maximum number of iterations used for training the FA 
# (Iterative methods only)
# (Defaults to 20000)
maxIteration = 20000


### Setting up the FA and experimenting with different training methods
############################################################################


# Create the FA object called, "funcApprox", and initialize it with the FA variables set above.
funcApprox = fa(numFeats, learningRate, minDelta, maxIteration)

################################################################
### Train the FA with the 4 methods described in the handout ###
### just uncomment the training method you wish to test. 	 ###
################################################################

## Gradient Descent
#funcApprox.train_GD()

## Least Squares
#funcApprox.train_LS(data)

## Recursive Least Squares
#funcApprox.train_RLS()

## Recursive Least Squares Version 2
#funcApprox.train_RLS2()


plotFA(funcApprox, data)
animPlotFA(funcApprox, data)


# Create the Locally Weighted FA object called, "funcApproxLocal", and initialize it with the FA variables set above.
funcApproxLocal = fa_lw(numFeats, minDelta, maxIteration)

#funcApproxLocal.train_LWLS(data)

#plotFA(funcApproxLocal, data)



