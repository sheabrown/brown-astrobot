# --------------------------------mlclass_emu.py------------------------------
# Written by Shea Brown using scikit-learn algorithms (shea-brown@uiowa.edu)
# This program is a first attempt at the EMU-WTF Data Challenge 1a, where
# we use a simple dense 2-layer neural network to classify images as either
# 'boring' or 'interesting', based on a training set developed by the WTF team
# ----------------------------------------------------------------------------
from sklearn import svm
import matplotlib.pyplot as plt
import scipy.misc
from neural_net import train, predict
import glob
import numpy as np

# We know the size of everything already, so we define the arrays for the data 
# (features) and the target classes for both training and classification sets
# ---------------------------------------------------------------------------
s=(354,4096) 		# Number of samples, number of features (64x64 images)
data=np.zeros(s) 	# Training set data
target=[]		# Array of classifications for the training set
ndata=np.zeros(s)	# Classification set data
truth=[]                # Array of true classifications for the 'unknown' data

# Read all the training images into an array of features (see data above)
# Read all the corresponding classes for each into a target array 
# ----------------------------------------------------------------------------
i=0
for filename in glob.glob('train/*.png'):
	im=scipy.misc.imread(filename)
	im=np.array(im)
	im=0.001*(im.flatten()-np.mean(im))
	if len(im) != 4096:
		im=np.resize(im,(4096))
	data[i,:]=im
	if 'B' in filename:
		target.append(0)
	if 'I' in filename:
                target.append(1)
	i=i+1

#print data[:,0]
target=np.array(target)
target=target.reshape((354,1))
print target.shape
# Read in the set of images to classify, along with their true
# classifications to check against
# ------------------------------------------------------------
i=0
for filename in glob.glob('class/*.png'):
        im=scipy.misc.imread(filename)
        im=np.array(im)
        im=0.001*(im.flatten()-np.mean(im))
        if len(im) != 4096:
                im=np.resize(im,(4096))
        ndata[i,:]=im
        if 'B' in filename:
                truth.append(0)
        if 'I' in filename:
                truth.append(1)
        i=i+1

# Train the classifier on the training set. The parameters came from another
# program where we used GridsearchCV() to optimize based on the training set  
# --------------------------------------------------------------------------
syn0=train(data, target,600)

# Try on the unknown images and store in an array of predictions (guesses)
# -------------------------------------------------------------------------
i=0
guesses=np.zeros(len(truth))
p=np.zeros(354)
for i in range(0,len(truth)-1):
	pred=predict(ndata[i,:].reshape(1,-1),syn0)
	p[i]=pred
	if pred >= 0.4:
		guesses[i]=1.0
	else:
		guesses[i]=0.0

print('The mean of p() is ',np.mean(p))
print('The rms of p() is ',np.sqrt(np.var(p)))
print('The 2sigma of p() is ',np.mean(p)+2*np.sqrt(np.var(p)))

# Compare this to the 'true' classifications. How many did we get right?
# -------------------------------------------------------------------------
truth=np.array(truth)
index=truth==1.0
cor=np.sum(guesses[index] == truth[index])
tot=len(truth[index])
print 'The estimator gave '+str(cor)+' interesting sources the correct classification, out of '+str(tot)+' total.'  
print 'That is a true positive rate of '+str(100.0*cor/tot)+'%'

index=truth==0.0
cor=np.sum(guesses[index] == truth[index])
tot=len(truth[index])
print 'The estimator gave '+str(cor)+' boring sources the correct classification, out of '+str(tot)+' total.'
print 'That is '+str(100.0*cor/tot)+'%'
print 'The false positive rate is '+str(100.0*(tot-cor)/tot)+'%'

plt.hist(p,bins=30)
plt.show()

