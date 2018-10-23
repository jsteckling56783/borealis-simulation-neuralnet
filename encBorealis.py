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
	total = 8000
	interval = 8000 // count
	X_data = []
	files = glob.glob ("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/images/small/*.JPG")
	j = 0
	for i, myFile in enumerate(files):
		if(i % interval == 0):
			if(j<count):
			    image = cv2.imread (myFile)
			    image = cv2.resize(image, (256, 96)) #scale down size (factor of 16 bc pooling 4x later)
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

	#applies autoencoder to a tensor and retruens decoded version
def autoencoder(pics):


	#encoder
	x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(pics)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	x = (Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	#x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	#x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x) #3, 10, 64
	#x = Flatten()(x) #stacks nodes an Nx1 dim fully connected layer # was 2560
	x = (Dense(16, activation='relu'))(x)	#basic nn layer (needs flatten first)
	#aut.add(Dropout(0.2))	#randomly drops proportions of dense nodes to avoid overfitting

	#decoder
	#x = (Dense(128, activation='relu'))(x)
	#x = tf.reshape(x, (800, 10, 32, 32))
	#x = (UpSampling2D(size=(2, 2)))(x)
	#x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	return x

#applies autoencoder to a tensor and retruens decoded version
def avgAutoencoder(pics):
		#encoder
	x = (Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(pics)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	x = (Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	x = (Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x)
	x = (Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	#x = (MaxPooling2D(pool_size=(2, 2), padding='same'))(x) #3, 10, 64			#next: remove kthis
	x = Flatten()(x) #stacks nodes an Nx1 dim fully connected layer # was 2560
	x = (Dense(500, activation='relu'))(x)	#basic nn layer (needs flatten first)
	#aut.add(Dropout(0.2))	#randomly drops proportions of dense nodes to avoid overfitting


	print("x in the encoder: ", x)

	#decoder
	x = (Dense(12288, activation='relu'))(x)
	x = (Dense(12288, activation='relu'))(x)
	x = Reshape((12, 32, 32))(x)
	#x = (UpSampling2D(size=(2, 2)))(x)											#also rm this
	x = (Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	x = (UpSampling2D(size=(2, 2)))(x)
	x = (Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same'))(x)
	return x


def makeAndTrain():
	picsTemlate = Input(shape = (96, 256, 1))
	autoenc = Model(picsTemlate, autoencoder(picsTemlate))
	autoenc.compile(loss='mean_squared_error', optimizer = RMSprop())
	autoenc.summary()

	# possibly use later
	# train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
	#                                                              train_data, 
	#                                                              test_size=0.2, 
	#                                                              random_state=13)

	#get and prepare train and test data
	sampleSize = 1000
	testSampSize = sampleSize * 4 // 5
	npPics = getData(sampleSize)
	npPics = npPics / 255
	np.random.shuffle(npPics)
	npTrain = npPics[:testSampSize] 
	npTest = npPics[testSampSize:]
	print('test sample size', testSampSize)
	print('entire size', npPics.shape)
	print('train size', npTrain.shape)
	print('test size', npTest.shape)

	#show(npPics, 7)
	#pics = tf.convert_to_tensor(npTrain, np.float32)

	# print('X_data shape:', npTrain.shape)
	# encoded = getEnc()(pics)
	# decoded = getDec()(getEnc()(pics))
	# encodedNp = tfToNp(encoded)
	# decodedNp = tfToNp(decoded)
	# print("pics dtype", pics.dtype)

	autoencoder_trainVals = autoenc.fit(npTrain, npTrain, batch_size=testSampSize // 5, epochs=5, verbose=1, validation_data=(npTest, npTest), shuffle=True)
	
	after = autoenc.predict(npTest)
	print('predicted size', after.shape)

	print("=================")
	print(" 1 weights: ", len(autoenc.get_layer(index=1).get_weights()))
	print(" 1 weights: ", autoenc.get_layer(index=1).get_weights()[0].size)
	print(" 3 weights: ", len(autoenc.get_layer(index=3).get_weights()))
	print(" 3 weights: ", autoenc.get_layer(index=3).get_weights()[0].size)
	print("    model info after training")
	print(" 1 weights: ", autoenc.get_layer(index=1).get_weights())
	print(" 1 weights: ", len(autoenc.get_layer(index=1).get_weights()))
	print(" 1 weights: ", autoenc.get_layer(index=1).get_weights()[0].size)
	print(" 3 weights: ", autoenc.get_layer(index=3).get_weights())
	print(" 3 weights: ", len(autoenc.get_layer(index=3).get_weights()))
	print(" 3 weights: ", autoenc.get_layer(index=3).get_weights()[0].size)
	print(" 5 weights: ", autoenc.get_layer(index=5).get_weights())
	print(" 5 weights: ", len(autoenc.get_layer(index=5).get_weights()))
	print(" 5 weights: ", autoenc.get_layer(index=5).get_weights()[0].size)

	saveLatent(autoenc, npTest)

	#show(after, 30)
	showSome(npTest, after)

	return autoenc

#pls = autoencoder(pics)90


def saveLatent(model, data):
	encoder = Sequential()
	for i in range(1, 7):
		encoder.add(model.get_layer(index=i))
	encoder.add(Flatten())
	latent = encoder.predict(data, verbose=1)
	np.savetxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent/latent5.csv", latent, delimiter=",")

def saveAveragedLatent(model, data):
	encoder = Sequential()
	for i in range(1, 10):
		encoder.add(model.get_layer(index=i))
	latent = encoder.predict(data, verbose=1)
	np.savetxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent/latent2.csv", latent, delimiter=",")
def saveLatent(model, data):
	encoder = Sequential()
	for i in range(1, 10):
		encoder.add(model.get_layer(index=i))
	latent = encoder.predict(data, verbose=1)
	np.savetxt("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/latent2.csv", latent, delimiter=",")

makeAndTrain()