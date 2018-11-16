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
from matplotlib.widgets import Slider, Button
import numpy as np
import cv2
import glob
#import gzip


#https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial

#specs of input images
IMG_COUNT = 1500
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
			    X_data.append(image) #green chanel
			if j == -99999:
				fig = plt.figure()
				title = 'Image from training. shape: ' + str(image.shape) + str(type(image))
				fig.suptitle(title, fontsize=11)
				plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
				print()
				print("==================")
				print()
				print("from image that is actually displaying not jank",  image[0:10:, 0])
			j+=1


	plt.show()
	images = np.array(X_data)
	print("xdata shape:", images.shape)
	return images

#depricated
def show(pics, number):
	fig = plt.figure(figsize=[2,2])
	fig.add_subplot(2, 1, 1)
	plt.imshow(cv2.cvtColor(pics[number], cv2.COLOR_BGR2RGB))
	#fig.add_subplot(2, 1, 2)
	#plt.imshow(pics[number, :, :, 1], cmap='Greens_r')
	plt.show()

def showSome(beforePics, afterPics, title):
	count = afterPics.shape[0]

	if(beforePics.shape[0] < count):
		count = beforePics

	interval = count//21
	j = interval

	fig = plt.figure(figsize=[13, 5])
	title += str(beforePics[0].shape) + ", " + str(type(beforePics[0])) + ", " + str(afterPics[0].shape) + ", " + str(type(afterPics[0]))
	fig.suptitle(title, fontsize=11)
	print("afterPics[j] type", afterPics[j].dtype)

	
	weirdly_high_red_indicies = afterPics[:, :, :, 2] > 150
	weirdly_high_green_indicies = afterPics[:, :, :, 0] > 253
	weirdly_high_blue_indicies = afterPics[:, :, :, 1] > 250
	afterPics[weirdly_high_red_indicies, 2] = 1.0
	afterPics[weirdly_high_green_indicies, 0] = 1.0
	afterPics[weirdly_high_blue_indicies, 1] = 1.0

	# fig = plt.figure()
	# fig.suptitle('blues', fontsize=11)
	# plt.hist(np.ndarray.flatten(afterPics[:, :, :, 0]), bins=255)
	# fig = plt.figure()
	# fig.suptitle('reds', fontsize=11)
	# plt.hist(np.ndarray.flatten(afterPics[:, :, :, 2]), bins=255)
	# fig = plt.figure()
	# fig.suptitle('purples', fontsize=11)
	# it = afterPics[:, :, :, 0].astype(float) + afterPics[:, :, :, 2].astype(float)
	# plt.hist(np.ndarray.flatten(it), bins=255)
	# plt.show()

	afterPics = afterPics.astype(np.uint8)
	print()
	print('============================')
	print()
	print("about to show pics")
	print("beforePics[j] shape", beforePics[j].shape)
	print("afterPics[j] shape", afterPics[j].shape)
	print("beforePics[j] type", type(beforePics[j]))
	print("afterPics[j] type", type(afterPics[j]))
	print("beforePics[j] type", beforePics[j].dtype)
	print("afterPics[j] type", afterPics[j].dtype)
	print("some from before pics displaying now better", beforePics[j, 0:10:, 0])
	print("some from after pics displaying jankly", afterPics[j, 0:10:, 0])
	#afterPics /= 255
	j = 100
	for i in range(5):
		fig.add_subplot(5, 4, (4*i)+1)
		plt.imshow(cv2.cvtColor(beforePics[j], cv2.COLOR_BGR2RGB))


		fig.add_subplot(5, 4, (4*i)+2)
		plt.imshow(cv2.cvtColor(afterPics[j], cv2.COLOR_BGR2RGB))

		j += interval

		switched = beforePics[j]
		fig.add_subplot(5, 4, (4*i)+3)
		plt.imshow(cv2.cvtColor(beforePics[j], cv2.COLOR_BGR2RGB))


		switched = afterPics[j]
		fig.add_subplot(5, 4, (4*i)+4)
		plt.imshow(cv2.cvtColor(afterPics[j], cv2.COLOR_BGR2RGB))

		j += interval
	enlarged = plt.figure(figsize=[13, 5])
	enlarged.add_subplot(2, 1, 2)
	enlarged.add_subplot
	plt.imshow(cv2.cvtColor(beforePics[j], cv2.COLOR_BGR2RGB), interpolation='bilinear')


	enlarged.add_subplot(2, 1, 1)
	plt.imshow(cv2.cvtColor(afterPics[j], cv2.COLOR_BGR2RGB), interpolation='bilinear')
	plt.show()


def printTf(tensor):
	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
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
def linSliderToNormal(bellCurveInfo):
	slider = bellCurveInfo[0]
	mean = bellCurveInfo[1]
	std = bellCurveInfo[2]

	# set values ouside of the 2-98 percentile range to the respective edge of the range to prevent extreme image features in slider demo
	slider = 0.01 if slider < 0.01 else (0.99 if slider > 0.99 else slider)
	norm.ppf
	return norm.ppf(slider, loc=mean, scale=std)

def getOutlierRemovedImage(img):
	outlier_red_indicies = img[:, :, 2] > 130
	outlier_green_indicies = img[:, :, 0] > 210
	outlier_blue_indicies = img[:, :, 1] > 170
	img[outlier_red_indicies, 2] = 1.0
	img[outlier_green_indicies, 0] = 1.0
	img[outlier_blue_indicies, 1] = 1.0
	return img.astype(np.uint8)

def getImgFromLatent(normalizedVals, latentToDistr):
	generatedImages = (np.matmul(normalizedVals, latentToDistr)).reshape((HEIGHT, WIDTH, 3))
	print("generatedImages shape", generatedImages.shape)

	generatedImages = generatedImages.astype(np.uint8)#since 0-255 color values must be integers to display
	return generatedImages

def wiggleSlightly(value):
	incr = np.random.uniform(low=-0.1, high=0.1)
	return value + incr

# takes in 1d nparray
def wiggleVectorSlightly(vector):
	return np.array(list(map(wiggleSlightly, vector)))

def pickRandomNormal(meanStd):
	return np.random.normal(loc=meanStd[0], scale=abs(meanStd[1]))

def pickRandom01Unif(ignore):
	return np.random.uniform(low=0.0, high=1.0)

#randomly chooses values for each latent PCA weight following the normal distribution and transforms this vector into is corresponding image
#TODO: might need bounds
#inten
def generateRandomStartImg(means, stds, latentToDistr):
		
		m = 1
		iterableDistr = np.column_stack((means, stds * m))
		u0 = np.ones_like(means)
		linSliderToNormal

		u0 = np.array(list(map(pickRandom01Unif, u0)))

		for i in range(200):
			#print("saving wiggled:", i)
			#w0 = np.ones_like(means)
			#w0 = stepForwardImgFromLatent(w0)
			#w0 = np.array(list(map(pickRandomNormal, iterableDistr)))


			iterableDistrUnif = np.column_stack((u0, means, stds))
			w0 = np.array(list(map(linSliderToNormal, iterableDistrUnif)))

			img0 = getImgFromLatent(w0, latentToDistr)
			#img0 = getOutlierRemovedImage(img0)

			blueOutliers = img0[:, :, 0]> 200
			img0[blueOutliers, 0] = 1.0
			greenOutliers = img0[:, :, 1]> 200
			img0[greenOutliers, 1] = 1.0
			redOutliers= img0[:, :, 2]> 150
			img0[redOutliers, 2] = 1.0

			img0 = img0.astype(np.uint8)


			

			show = (i%50==0)
			if show:
				fig = plt.figure(figsize=[8, 4])
				fig.suptitle('Generated from random sampling sequence from PCA ' + str(i), fontsize=12)
				plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB), interpolation='nearest')

			save = True
			if save:
				path = 'genSeqImgsOrd_' + str(i) + '.png'
				#plt.imsave(path, cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))


			u0 = stepForwardImgFromLatent(u0)

#takes 1d np array
def stepForwardImgFromLatent(w0):
	w0 = np.array(list(map(wiggleSlightly, w0)))
	return w0


def runSliders(means, stds, latentToDistr):
	dims = means.size
	sliderVals = (np.ones_like(means))*0.3
															#todo: pass data without making so many copies
	iterableDistr = np.column_stack((sliderVals, means, stds))

	normalizedVals = np.array(list(map(linSliderToNormal, iterableDistr)))
  
	#displays an image, each with one PC turned up by a standard deviaion
	for i in range(0):
		alteredMeans = means
		alteredMeans[i] += stds[i]
		demoImg = getImgFromLatent(alteredMeans, latentToDistr)
		fig = plt.figure(figsize=[5, 2])
		fig.suptitle('This is regenerated from the average image', fontsize=12)
		#plt.axis([0, 120, 0, 1])
		blueOutliers = demoImg[:, :, 0]> 253
		demoImg[blueOutliers, 0] = 1.0
		greenOutliers = demoImg[:, :, 1]> 253
		demoImg[greenOutliers, 1] = 1.0
		redOutliers= demoImg[:, :, 2]> 150
		demoImg[redOutliers, 2] = 1.0


		demoImg = demoImg.astype(np.uint8)
		print()
		print('========================================================')
		print()
		print("demoimg shape: ", demoImg.shape)
		print("some demoimg values: ", demoImg[0, 0])

		plt.imshow(cv2.cvtColor(demoImg, cv2.COLOR_BGR2RGB), interpolation='nearest')


	#f0=0.5
	#sliderax = plt.axes([0.25, 0.95, 0.1, 0.03], facecolor='lightgoldenrodyellow')
	#plt.axis('off')
	#slider = Slider(sliderax, 'Dim 1', 0, 1, valinit=f0)

	#resetax = plt.axes([0.8, 0.95, 0.1, 0.04])
	#button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

	#button.on_clicked(updateFromSliders)

	# t = np.arange(0.0, 1.0, 0.001)
	# amp_0 = 5
	# freq_0 = 3

	# amp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg='lightgoldenrodyellow')
	# amp_slider = Slider(amp_slider_ax, 'Amp', 0.1, 10.0, valinit=amp_0)

	# # Draw another slider
	# freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow')
	# freq_slider = Slider(freq_slider_ax, 'Freq', 0.1, 30.0, valinit=freq_0)






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
	print("in getdst latent shape:", latent.shape)
	means = getColMeans(latent)
	stds = getColStds(latent)

	print("in getdst std shape:", stds.shape)
	stats = np.column_stack((means, stds))
	print()
	print('========================================================')
	print()
	print("stats (mean, std):", stats)
	return stats


#factor into svd decomposition, then reconstruct. should be the same since no principal compnents are eliminated/selected
#should work with form (count, height*width) tensor if using an image
def getSVDRankRApprox(x, rank, runTraining):



	if runTraining:
		print("image set shape for getSVDRankRApprox:", x.shape)
		print("about to train pca again")
		(u, s, vt) = np.linalg.svd(x, full_matrices=False)
		uA = u[:, :rank]
		sA = s[:rank]
		sDA = np.diag(sA) 
		vtA = vt[:rank]
		print("    original svd shapes: ", u.shape, s.shape, "( -> square", vt.shape)
		print("reduced rank svd shapes: ", uA.shape, sDA.shape, vtA.shape)
		svtA = np.matmul(sDA, vtA)
		#np.savetxt("latent/120pca_uA_colored.csv", uA, delimiter=",")
		#np.savetxt("latent/120pca_uAWeights_colored.csv", svtA, delimiter=",")


	else:
		print("reading pca output from training earlier")
		uA = np.genfromtxt("latent/120pca_uA_colored.csv", delimiter=",")
		svtA = np.genfromtxt("latent/120pca_uAWeights_colored.csv", delimiter=",")
		print("reduced rank svd shapes: ", uA.shape, svtA.shape)

		print("read the files")
	
	#np.savetxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent/pcaLatent120.csv", uA, delimiter=",")
	stats = getLatentDistribution(uA)
	latentWeights = svtA
	#plotSimilarLatent(stats)
	#plot2Series(uA[:, 3], uA[:, 4])
	runSliders(stats[:,0], stats[:,1], latentWeights)
	
	generateRandomStartImg(stats[:,0], stats[:,1], latentWeights)
	plt.show()


	x_approx = np.matmul(uA, svtA)

	return np.ceil(x_approx) #get ineger 0-255 color values


def pca():
	np.set_printoptions(threshold=100, edgeitems=5)
	n = IMG_COUNT #number of images
	(w, h) = (WIDTH, HEIGHT)
	picsViz = getDataResize(n, (w, h)) #format that can be image rendered
	print("picsViz shape:", picsViz.shape)
	pics = picsViz.reshape((n, w*h*3)) #flattened: suitible for PCA
	print("pics shape:", pics.shape)

	x = getSVDRankRApprox(pics, 120, False)
	print("type:", x.dtype)
	#x = np.array(x, dtype=np.uint8)

	print("approx did")
	print("error rate: ", getLoss(x, pics))
	xViz = x.reshape(n, h, w, 3)

	
	xViz = xViz.astype(np.uint8)
	showSome(picsViz, xViz, 'Original vs svd approximated')

	print("done")

pca()

