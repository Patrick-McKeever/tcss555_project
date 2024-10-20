# !/usr/bin/python3

import argparse
import os
import sys
import csv
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, BatchNormalization, Activation, AveragePooling2D, Dropout, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras import regularizers, optimizers, models, layers

def check_valid_directory(path):
	if not os.path.isdir(path):
		raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
	return path

def check_valid_modality(s):
	if s not in set(["image", "text", "like"]):
		raise argparse.ArgumentTypeError(
			f"'{s}' is not a valid modality, should be one of"
			f"'image', 'text', or 'like'."
		)
	return s

def get_profile_rows(in_dir):
	prof_path = os.path.join(in_dir, "profile", "profile.csv")
	if not os.path.isfile(prof_path):
		print(f"{prof_path} not found, exiting")
		exit(1)

	rows = []
	with open(prof_path, mode='r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)
	return rows
			

def classify(in_dir, out_dir):
	profile_rows = get_profile_rows(in_dir)
	for row in profile_rows:
		outstr = f"""<user
			id=\"{row['userid']}\"
			age_group=\"xx-24\"
			gender=\"female\"
			extrovert=\"3.486857895\"
			neurotic=\"2.732424\"
			agreeable=\"3.583904211\"
			conscientious=\"3.445616842\"
			open=\"3.908690526\"
		/>"""
	
		outpath = os.path.join(out_dir, f"{row['userid']}.xml")
		with open(outpath, 'w+') as of:
			of.write(outstr)
	

def resize_image(image_matrix, targ_h, targ_w):
	h, w, channels = image_matrix.shape
	new_image_matrix = np.zeros((targ_h, targ_w, channels), dtype='float32')
	h_scaling_factor = h / targ_h
	w_scaling_factor = w / targ_w
	
	for row_ind in range(targ_h):
		row_nn = row_ind * h_scaling_factor
		for col_ind in range(targ_w):
			col_nn = col_ind * w_scaling_factor
			new_image_matrix[row_ind, col_ind] = image_matrix[row_nn, col_nn]
	return new_image_matrix
			
def read_image(in_dir, user_id):
	image_path = os.path.join(in_dir, 'image', f'{user_id}.jpg')
	image = cv2.imread(image_path).astype('float32')

	# Normalize color values before returning
	return image / 255.0
	

def train_image_model(in_dir, out_dir):
	profiles = get_profile_rows(in_dir)
	genders = [profile['gender'] for profile in profiles]
	
	# First, compute min height / width of all images for resizing
	# purposes. Do not store all images in memory; if we did so,
	# we'd exhaust RAM.
	min_height = 999999
	min_width = 999999
	images = {
		"filename": [],
		"gender": []
	}

	for profile in profiles:
		image_matrix = read_image(in_dir, profile['userid'])
		min_height = min(min_height, image_matrix.shape[0])
		min_width = min(min_width, image_matrix.shape[1])

		image_path = f"{profile['userid']}.jpg"
		images["filename"].append(image_path)
		# Gender is already a scalar value, 0 /1.
		# (I forgot if 1 is male or female.)
		images["gender"].append(profile["gender"])
	
	images_df = pd.DataFrame(images)

	datagen = ImageDataGenerator(rescale=1./255., validation_split=0.1)
	train_generator = datagen.flow_from_dataframe(
		dataframe = images_df,
		directory = os.path.join(in_dir, "image"),
		x_col="filename",
		y_col="gender",
		subset="training",
		batch_size=32,
		seed=42,
		shuffle=True,
		class_mode="binary",
		target_size=(96, 96)
	)

	validation_generator = datagen.flow_from_dataframe(
		dataframe = images_df,
		directory = os.path.join(in_dir, "image"),
		x_col="filename",
		y_col="gender",
		subset="validation",
		batch_size=32,
		seed=42,
		shuffle=True,
		class_mode="binary",
		target_size=(96, 96)
	)

	# The model structure is taken from https://github.com/oarriaga/face_classification
	# (code accompanying Arriaga et al [2017]), except I add some fully connected 
	# layers at the end.
	model = Sequential()
	model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
	                        name='image_array', input_shape=(96,96,3)))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(.5))
	
	model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(.5))
	
	model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(.5))
	
	model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(.5))
	
	model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Convolution2D(
	    filters=1, kernel_size=(3, 3), padding='same'))
	model.add(GlobalAveragePooling2D())
	model.add(Dense(32))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	print(f"Model summary: {model.summary()}")

	model.compile(optimizer='adam', loss='binary_crossentropy', 
		metrics=['accuracy'])
	
	step_size_train = train_generator.n // train_generator.batch_size
	step_size_validate = validation_generator.n // validation_generator.batch_size

	checkpt_path = os.path.join(out_dir, 'model_checkpoint.h5')
	checkpoint = keras.callbacks.ModelCheckpoint(
		checkpt_path,
		monitor='val_loss',
		save_best_only=True,
		save_weights_only=False,
		mode='auto',
		verbose=1
	)

	early_stopping = keras.callbacks.EarlyStopping(
    	monitor='val_loss',
    	patience=5,
    	restore_best_weights=True
	)


	history = model.fit(
		train_generator,
		steps_per_epoch=step_size_train,
		validation_data=validation_generator,
		validation_steps=step_size_validate,
		epochs=50,
		callbacks=[checkpoint, early_stopping]
	)

	val_loss, val_acc = model.evaluate(
		validation_generator,
		steps=step_size_validate
	)

	print(f"Achieved val loss {val_loss}, val acc {val_acc}")
	return (val_loss, val_acc, history)
	
			
	

def train_text_model(in_dir, out_dir):
	pass
			
def train_like_model(in_dir, out_dir):
	pass

def train_model(in_dir, out_dir, modality):
	if modality == "image":
		return train_image_model(in_dir, out_dir)
	elif modality == "text":
		return train_text_model(in_dir, out_dir)
	elif modality == "like":
		return train_like_model(in_dir, out_dir)
	
	print(f"Error: {modality} is not 'image', 'text', or 'like'")
	exit(1)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Check input and output directory paths.')

	parser.add_argument('-i', '--input_dir', required=True, type=check_valid_directory,
						help='Path to the input directory')
	parser.add_argument('-o', '--output_dir', required=True, type=check_valid_directory,
						help='Path to the output directory')
	parser.add_argument('-t', '--train', required=False, type=check_valid_modality, 
						help='Model to train, should be image, text, or like')
	


	args = parser.parse_args()
	if args.train:
		print(f"Training {args.train} model")
		train_model(args.input_dir, args.output_dir, args.train)

	else:
		print(f"Classifying")
		classify(args.input_dir, args.output_dir)
