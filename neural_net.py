# ============================================================================
# This is a simple neural network (really just a linear logistic classifier) 
# with no frills, meant to train a vector of weights (no bias in this one),
# used in Astrophysical Machine Learning at the University of Iowa 
# https://astrophysicalmachinelearning.wordpress.com/ taught by Shea Brown
# ============================================================================
import numpy as np

# sigmoid function
# -----------------------
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# Function to train the neural net
# --------------------------------
def train(X,y,N=10000):
	np.random.seed(1)
	print X.shape	
	print y.shape
	# initialize weights randomly with mean 0
	W = 2*np.random.random((len(X[0]),1)) - 1
	print len(W)
	for iter in xrange(N):

    		# forward propagation
		l0 = X
	    	l1 = nonlin(np.dot(l0,W))

		# how much did we miss?
    		l1_error = y - l1

		# multiply how much we missed by the 
    		# slope of the sigmoid at the values in l1
    		l1_delta = l1_error * nonlin(l1,True)
    		# update weights
    		W += np.dot(l0.T,l1_delta)
	
	return W 

# Function to make a prediction with user-defined weights W
# ----------------------------------------------------------
def predict(data,W):
	out=nonlin(np.dot(data,W))
	return out

