import os
import numpy as np
import pandas as pd
import cv2 as cv

folders = ['center', 'curves', 'straighten', 'default', 'center_jungle', 'curves_jungle', 'straighten_jungle']

samples = pd.DataFrame()
sample_list = []

for folder in folders:
	sample = pd.read_csv('./data_both/' + folder + '/driving_log.csv', usecols=[0, 1, 2, 3], index_col=None, header=0)
	sample['folder'] = folder
	print('LENGTH OF '+ folder + ' DATAFRAME :: ' + str(len(sample)))
	sample_list.append(sample)

print('LENGTH OF SAMPLE LIST')
print(len(sample_list))

samples = pd.concat(sample_list)

print('LENGTH OF SAMPLE DATAFRAME')
print(len(samples))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def flip(image, angle):
	return np.fliplr(image), -angle


def shift(image, angle):
	translation_x = np.random.randint(0, 100) - 50

	angle += translation_x * 0.004

	traslation_y = np.random.randint(0, 40) - 20

	translation_matrix = np.float32([[1, 0, translation_x], [0, 1, traslation_y]])

	image = cv.warpAffine(image, translation_matrix, (320, 160))	

	return image, angle


def bright(image):
	image = np.array(image, dtype=np.float64)

	brightness = np.random.uniform() + 0.5

	image[:, :, 2] += brightness

	image[:,:,2][image[:,:,2] > 255] = 255

	image = np.array(image, dtype=np.uint8)

	return image


def augment(image, angle, mode):
	augments = ['flip', 'shift', 'bright', 'none']
	
	if mode == 'train':
		augment = np.random.choice(augments)
	else:
		augment = augments[-1]

	if augment == 'flip':
		image, angle = flip(image, angle)
	elif augment == 'shift':
		image, angle = shift(image, angle)
	elif augment == 'bright':
		image = bright(image)

	# Crop out uneccessary information
	image = image[50:140]

	# Resize image to the shape expected ny the model
	image = cv.resize(image, (200, 66), interpolation=cv.INTER_AREA)

	return image, angle


def generator(samples, mode='train', batch_size=32):
	no_samples = len(samples)

	while True:
		shuffle(samples)

		for start in range(0, no_samples, batch_size):
			images = []
			steering_angles = []

			for index in range(start, start + batch_size):
				if index < no_samples:
						folder_name = samples.iat[index, 4]

						image_name = lambda x: './data_both/' + folder_name + '/IMG/' + samples.get_value(index, x, takeable=True).split('/')[-1]
						angle_name = lambda x, y: float(samples.get_value(index, x, takeable=True)) + y

						cameras = ['center', 'left', 'right']
						corrections = [0, 0.2, -0.2]
						
						if mode == 'train':
							cam = np.random.choice(cameras)
						else:
							cam = cameras[0]

						image = cv.imread(image_name(cameras.index(cam)))
						image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

						angle = angle_name(3, corrections[cameras.index(cam)])

						image, angle = augment(image, angle, mode)

						images.append(np.reshape(image, (66, 200, 3) ))
						steering_angles.append(angle)


			# trim image to only see section with road
			x_train = np.array(images)
			y_train = np.array(steering_angles)

			yield shuffle(x_train, y_train)


# compile and train the model using the generator function 
train_generator = generator(train_samples, mode='train', batch_size=64)
validation_generator = generator(validation_samples, mode='validation', batch_size=64)

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras import optimizers
import matplotlib.pyplot as plt

channel, row, column = 3, 66, 200

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, column, channel),
        output_shape=(row, column, channel)))

# model.add(Cropping2D(cropping=((70, 20), (0, 0)), input_shape=(row, column, channel)))

# convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Dropout(0.2))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Dropout(0.2))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Dropout(0.2))

# fully connected layers
model.add(Flatten())

model.add(Dense(100, activation='elu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation='elu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='elu'))

model.add(Dense(1))

model.summary()
model.compile(loss="mse", optimizer=optimizers.Adam(lr=1e-04))

history_object = model.fit_generator(train_generator, samples_per_epoch=3*len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=40, verbose=1)

# save_model(model)
model.save('model_nvidia_both.h5')

with open("model_nvidia_both.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model Saved.")

# print the keys contained in the history object
# print(history_object.keys())

# plot the training and validation loss for each epoch 
# plt.plot(history_object.history['loss'])
# plt.plot(history_object['val_loss'])

# plt.title('Model Mean Squared Error Loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')

# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()