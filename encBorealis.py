import keras
from keras import datasets
from keras.layers import Input,  Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model 
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob
#import gzip


#https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial


def getData(count):
	X_data = []
	files = glob.glob ("C:/Users/Jessie Steckling/Documents/Code/GitHub/borealis-simulation-neuralnet/images/small/*.JPG")
	j = 0
	for i, myFile in enumerate(files):
		if(i % 10 == 0):
			if(j<count):
			    image = cv2.imread (myFile)
			    image = cv2.resize(image, (160, 48)) #scale down size (factor of 16 bc pooling 4x later)
			    cv2.resize
			    X_data.append (image[:, :, :2]) #blue and green chanels
			j+=1

	return np.array(X_data)

def show(pics, number):
	fig = plt.figure(figsize=[2,2])
	fig.add_subplot(2, 1, 1)
	plt.imshow(pics[number, :, :, 0], cmap='Blues_r')
	fig.add_subplot(2, 1, 2)
	plt.imshow(pics[number, :, :, 1], cmap='Greens_r')
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


def autoencoder(pics):
	aut = keras.Sequential()

	#encoder
	aut.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
	aut.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	aut.add(Conv2D(16, kernel_size=(4, 4), activation='relu', padding='same'))
	aut.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	aut.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
	aut.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	aut.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
	aut.add(MaxPooling2D(pool_size=(2, 2), padding='same')) #3, 10, 64
	#aut.add(Flatten()) #stacks nodes an Nx1 dim fully connected layer # was 2560
	#aut.add(Dense(20, activation='relu')) 	#basic nn layer (needs flatten first)
	#aut.add(Dropout(0.2))	#randomly drops proportions of dense nodes to avoid overfitting

	#decoder
	aut.add(UpSampling2D(size=(2, 2)))
	aut.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
	aut.add(UpSampling2D(size=(2, 2)))
	aut.add(Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same'))
	aut.add(UpSampling2D(size=(2, 2)))
	aut.add(Conv2D(16, kernel_size=(4, 4), activation='relu', padding='same'))
	aut.add(UpSampling2D(size=(2, 2)))
	aut.add(Conv2D(2, kernel_size=(3, 3), activation='relu', padding='same'))
	after = aut(pics)
	return after

npPics = getData(100)
#show(npPics, 7)
print('X_data shape:', npPics.shape)
pics = tf.convert_to_tensor(npPics, np.float32)

# encoded = getEnc()(pics)
# decoded = getDec()(getEnc()(pics))
# encodedNp = tfToNp(encoded)
# decodedNp = tfToNp(decoded)

print(pics.dtype)


pls = autoencoder(pics)
picsTemlate = Input(shape = (48, 160, 2))
autoencoder = Model(picsTemlate, autoencoder(picsTemlate))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()



# train_data = pics / np.max(pics)
# train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
#                                                              train_data, 
#                                                              test_size=0.2, 
#                                                              random_state=13)
# autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=20,epochs=2,verbose=1,validation_data=(valid_X, valid_ground), shuffle=True)
