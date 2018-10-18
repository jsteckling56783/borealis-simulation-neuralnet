import keras
from keras import datasets, regularizers
from keras.layers import Input,  Dense, Dropout, Flatten, Reshape, Embedding, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
#import gzip


#https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial

PARAM_SIZE = 80


def getData(count):
	return getDataResize(count, (256, 96))

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
	plt.show()

def showEnc(pics, number):
	fig = plt.figure(figsize=[12, 5])
	for i in range(64):
		fig.add_subplot(6, 12, i+1).set_yticklabels([])
		plt.imshow(pics[number, :, :, i], cmap='Greys')
	plt.show()

def showFlattenedButUseRarelyBecauseExpensive(pics, number): #takes in a 4d tensor
	model = keras.Sequential()
	model.add(Flatten())
	flat = model(pics)
	d1Many = tfToNp(flat)
	print('d1many', d1Many.shape)
	d1 = d1Many[number]
	print('d1', d1.shape)
	for i in range(4):
		d1 = np.dstack((d1, d1))
	d2=d1
	print('d2', d2.shape)
	d2 = d2[0]
	print('d2', d2.shape)
	d2 = d2.T
	print('d2', d2.shape)
	fig = plt.figure(figsize=[10, 3])
	fig.add_subplot(1, 1, 1).set_yticklabels([])
	plt.imshow(d2[:, :1000], cmap="Greens_r", interpolation='nearest')
	plt.show()	
	print(d1)

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
	print(u.shape)
	print(sD.shape)
	print(vt.shape)
	x_approx = np.matmul(np.matmul(u, sD), vt)  # flattened. exact, not approximation since all columns kept
	return x_approx

#factor into svd decomposition, then reconstruct. should be the same since no principal compnents are eliminated/selected
#should work with form (count, height*width) tensor if using an image
def getSVDRankRApprox(x, rank):

	(u, s, vt) = np.linalg.svd(x, full_matrices=True)
	uA = u[:, :rank]
	sA = s[:rank]
	sDA = np.diag(sA)
	#plotSingularValues(s)
	vtA = vt[:rank]
	print(uA.shape)
	print(sDA.shape)
	print(vtA.shape)
	x_approx = np.matmul(np.matmul(uA, sDA), vtA)  # flattened. exact, not approximation since all columns kept
	return x_approx


def pca():
	np.set_printoptions(threshold=100, edgeitems=10)
	n = 1500 #number of images
	(w, h) = (256, 64)
	picsViz = getDataResize(n, (w, h)) #format that can be image rendered
	pics = picsViz.reshape((n, w*h)) #flattened: suitible for PCA
	print("pics", pics.shape)
	x = getSVDRankRApprox(pics, 100)
	print("approx did")
	print("error rate: ", getLoss(x, pics))
	xViz = x.reshape(n, h, w, 1)
	print("xviz", xViz.shape)
	print("picsViz", picsViz.shape)

	showSome(picsViz, xViz)

	print("done")

pca()

#makeAndTrain()