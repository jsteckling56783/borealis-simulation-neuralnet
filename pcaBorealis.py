import keras
from keras import datasets, regularizers
from keras.layers import Input,  Dense, Dropout, Flatten, Reshape, Embedding, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import tensorflow as tf
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import cv2
import glob
#import gzip


#https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial

#specs of input images
IMG_COUNT = 4000
WIDTH = 256
HEIGHT = 64


def getData(count):
	return getDataResize(count, (256, 64))

def getDataResize(count, size):
	total = 8000
	interval = 8000 // count
	X_data = []
	files = glob.glob ("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/images/small/*.JPG")
	j = 0
	for i, myFile in enumerate(files):
		if(i % interval == 0):
			if(j<count):
			    image = cv2.imread (myFile)
			    image = cv2.resize(image, size) 
			    cv2.resize
			    X_data.append (image[:, :, 1:2]) #blue and green chanels
			j+=1

	return np.array(X_data)



def show(pics, number):
	fig = plt.figure(figsize=[2,2])
	fig.add_subplot(2, 1, 1)
	plt.imshow(pics[number, :, :, 0], cmap='Blues_r')
	#fig.add_subplot(2, 1, 2)
	#plt.imshow(pics[number, :, :, 1], cmap='Greens_r')
	plt.show()

def showSome(beforePics, afterPics):
	count = afterPics.shape[0]

	if(beforePics.shape[0] < count):
		count = beforePics

	interval = count//21

	fig = plt.figure(figsize=[13, 5])
	j = interval
	for i in range(5):
		fig.add_subplot(5, 4, (4*i)+1)
		plt.imshow(beforePics[j, :, :, 0], cmap='Greens_r')

		fig.add_subplot(5, 4, (4*i)+2)
		plt.imshow(afterPics[j, :, :, 0], cmap='Blues_r')

		j += interval

		fig.add_subplot(5, 4, (4*i)+3)
		plt.imshow(beforePics[j, :, :, 0], cmap='Greens_r')

		fig.add_subplot(5, 4, (4*i)+4)
		plt.imshow(afterPics[j, :, :, 0], cmap='Blues_r')

		j += interval
	enlarged = plt.figure(figsize=[13, 5])
	enlarged.add_subplot(2, 1, 2)
	enlarged.add_subplot
	plt.imshow(beforePics[j, :, :, 0], cmap='Greens_r')

	enlarged.add_subplot(2, 1, 1)
	plt.imshow(afterPics[j, :, :, 0], cmap='Blues_r')
	plt.show()


def showEnc(pics, number):
	fig = plt.figure(figsize=[12, 5])
	for i in range(64):
		fig.add_subplot(6, 12, i+1).set_yticklabels([])
		plt.imshow(pics[number, :, :, i], cmap='Greys')
	plt.show()


def printTf(tensor):
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	print(sess.run(tf.Print(encoded, [encoded], message='this be the tensor')))
	sess.close()

def tfToNp(tensor):
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	ret = tensor.eval()
	sess.close()
	return ret

def plot2Series(a, b):
	plt.figure()
	plt.plot(a, b, "r.")
	plt.show()

#maps a slider value [0, 1] to a value on the bell curve between -3 and 3 standard deviatons
# -0.5 maps to 25th percentile, 0 to median, 0.5 to 75th percentile, ect.
def linSliderToNormal(slider, mean, std):

	# set values ouside of the 2-98 percentile range to the respective edge of the range to prevent extreme image features in slider demo
	slider = 0.01 if slider < 0.01 else (0.99)
	return norm.ppf(slider, loc=mean, scale=std)



def runSliders(means, stds, latentToDistr):
	dims = means.size
	sliderVals = (np.ones_like(means))*0.5

	demoImg = (np.matmul(sliderVals, latentDistr)).reshape((HEIGHT, WIDTH))

	amp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
	amp_slider = Slider(amp_slider_ax, 'Amp', 0.1, 10.0, valinit=amp_0)

	# Draw another slider
	freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
	freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0)

	plt.figure()
	plt.imshow(demoImg, cmap='Greens_r')
	plt.show();



def plotSingularValues(s):
	sSeries = s.reshape(s.size, 1)[:1000]
	sIndex = np.ones_like(sSeries)
	for i, s in enumerate(sIndex):
		sIndex[i] = i+1
	#print(sIndex)
	#print(sSeries)

	plt.plot(sIndex, sSeries, "r.")

	#cumulaive
	plt.figure()
	sCum = np.copy(sSeries)
	for i, s in enumerate(sCum):
		if i>0:
		 sCum[i] += sCum[i-1]

	plt.plot(sIndex, sCum, "r.")
	plt.show()

def getLoss(actual, expected):
	return np.linalg.norm(actual-expected)/np.linalg.norm(expected)

#factor into svd decomposition, then reconstruct. should be the same since no principal compnents are eliminated/selected
#should work with form (count, height*width) tensor if using an image
def getSVDecomposedVersion(x):

	(u, s, vt) = np.linalg.svd(x, full_matrices=True)
	sD = np.diag(s)
	#print(u.shape)
	#print(sD.shape)
	#print(vt.shape)
	x_approx = np.matmul(np.matmul(u, sD), vt)  # flattened. exact, not approximation since all columns kept
	return x_approx

def getColMeans(valArray):
	return np.mean(valArray, axis=0)


def getColStds(valArray):
	return  np.std(valArray, axis=0)

#distr is a 2d array of n features by mean, stddev
def plotSimilarLatent(distr):
	feat1 = np.random.normal(loc=distr[2, 0], scale=distr[2, 1], size=1000)
	feat2 = np.random.normal(loc=distr[2, 0], scale=distr[3, 1], size=1000)
	plot2Series(feat1, feat2)


#latent parameter: 2d array of n photos x r latent dimensions
#returns 1-d array of tuples (mean, std)
def getLatentDistribution(latent):
	means = getColMeans(latent)
	stds = getColStds(latent)
	stats = np.column_stack((means, stds))
	print("stats:", stats)
	print("    stats shape:", stats.shape)
	return stats


#factor into svd decomposition, then reconstruct. should be the same since no principal compnents are eliminated/selected
#should work with form (count, height*width) tensor if using an image
def getSVDRankRApprox(x, rank):

	runTraining = True

	if runTraining:
		(u, s, vt) = np.linalg.svd(x, full_matrices=True)
		

	#else:
	#	u = np.genfromtxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent/pcaLatent120.csv", delimiter=",")

	uA = u[:, :rank]
	#randomuA = np.random.uniform(low=-1, high=1, size=uA.size).reshape(uA.shape)
	sA = s[:rank]
	sDA = np.diag(sA)
	#plotSingularValues(s)
	vtA = vt[:rank]
	print("    original svd shapes: ", u.shape, s.shape, "( -> square", vt.shape)
	print("reduced rank svd shapes: ", uA.shape, sDA.shape, vtA.shape)
	x_approx = np.matmul(np.matmul(uA, sDA), vtA)  # flattened. exact, not approximation since all columns kept
	np.savetxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent/pcaLatent120.csv", uA, delimiter=",")
	stats = getLatentDistribution(uA)
	latentWeights = np.matmul(sDA, vtA)
	#plotSimilarLatent(stats)
	plot2Series(uA[:, 5], uA[:, 6])
	runSliders(stats[0], stats[1], latentWeights)

	return x_approx


def pca():
	np.set_printoptions(threshold=100, edgeitems=8)
	n = IMG_COUNT #number of images
	(w, h) = (WIDTH, HEIGHT)
	picsViz = getDataResize(n, (w, h)) #format that can be image rendered
	pics = picsViz.reshape((n, w*h)) #flattened: suitible for PCA
	print("pics", pics.shape)
	x = getSVDRankRApprox(pics, 120)
	print("approx did")
	print("error rate: ", getLoss(x, pics))
	xViz = x.reshape(n, h, w, 1)
	showSome(picsViz, xViz)

	print("done")

pca()


#makeAndTrain()