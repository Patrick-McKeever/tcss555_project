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

# Declare this globally to prevent reloading.
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


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
	

def extract_face(image, target_size=(64,64)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) != 1:
        return None
    
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, target_size)
    return face_resized
	
def get_base_cnn(input_shape=(64,64,1)):
	model = models.Sequential([
		layers.Conv2D(32, (3, 3), input_shape=(64, 64, 1)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		
		layers.Conv2D(32, (3, 3)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		layers.MaxPooling2D((2, 2)),
		
		layers.Conv2D(64, (3, 3)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		
		layers.Conv2D(64, (3, 3)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		layers.MaxPooling2D((2, 2)),
		
		layers.Conv2D(64, (3, 3)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		
		layers.Conv2D(64, (3, 3)),
		layers.BatchNormalization(),
		layers.ReLU(),
		layers.Dropout(0.3),
		layers.MaxPooling2D((2, 2)),
		
		layers.Flatten(),
		
		layers.Dense(64),
		layers.BatchNormalization(),
		layers.ReLU()
	])
	return model

def get_faces_df(in_dir):
	profiles = get_profile_rows(in_dir)
	images = {
		"filename": [],
		"gender": [],
		"age": [],
		"ope": [],
		"con": [],
		"ext": [],
		"agr": [],
		"neu": []
	}

	target_size = (64, 64)
	face_dir = os.path.join(in_dir, "faces")
	regen = not os.path.exists(face_dir)
	if regen:
		print(f"Regenerating faces in {face_dir}")
		os.makedirs(face_dir)

	for profile in profiles:
		image_basename = f"{profile['userid']}.jpg"
		image_path = os.path.join(in_dir, "image", image_basename)
		face_path = os.path.join(face_dir, image_basename)
		image_matrix = cv2.imread(image_path)

		face_exists = False
		if regen:
			face_matrix = extract_face(image_matrix, target_size)
			if face_matrix is not None:
				cv2.imwrite(face_path, face_matrix)
				face_exists = True
		else:
			face_exists = os.path.exists(face_path)

		if face_exists:
			images["filename"].append(image_basename)
			images["gender"].append(profile["gender"])
			images["ope"].append(float(profile["ope"]))
			images["con"].append(float(profile["con"]))
			images["ext"].append(float(profile["ext"]))
			images["agr"].append(float(profile["agr"]))
			images["neu"].append(float(profile["neu"]))

			age_val = float(profile["age"])

			age = "xx-24"
			if age_val > 24 and age_val <= 34:
				age = "25-34"
			elif age_val > 34 and age_val <= 49:
				age = "35-49"
			elif age_val > 49:
				age = "50-xx"

			images["age"].append(age)
	
	faces_df = pd.DataFrame(images)
	return faces_df

def get_img_gens(face_df, face_dir, attr, class_mode):
	print(face_df)

	datagen = ImageDataGenerator(rescale=1./255., validation_split=0.1)
	train_generator = datagen.flow_from_dataframe(
		dataframe = face_df,
		directory = face_dir,
		x_col="filename",
		y_col=attr,
		subset="training",
		batch_size=32,
		seed=42,
		shuffle=True,
		class_mode=class_mode,
		target_size=(64,64),
		color_mode="grayscale"
	)

	validation_generator = datagen.flow_from_dataframe(
		dataframe = face_df,
		directory = face_dir,
		x_col="filename",
		y_col=attr,
		subset="validation",
		batch_size=32,
		seed=42,
		shuffle=True,
		class_mode=class_mode,
		target_size=(64,64),
		color_mode="grayscale"

	)

	return (train_generator, validation_generator)

def train_model(model, train_gen, val_gen, checkpt_path):
	step_size_train = train_gen.n // train_gen.batch_size
	step_size_validate = val_gen.n // val_gen.batch_size

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
		train_gen,
		steps_per_epoch=step_size_train,
		validation_data=val_gen,
		validation_steps=step_size_validate,
		epochs=50,
		callbacks=[checkpoint, early_stopping]
	)

	val_loss, val_acc = model.evaluate(
		val_gen,
		steps=step_size_validate
	)

	return (val_loss, val_acc)


def train_image_models(in_dir, out_dir):
	face_dir = os.path.join(in_dir, "faces")
	print(f"Face dir {face_dir}")
	face_df = get_faces_df(in_dir)
	(gender_train_gen, gender_val_gen) = get_img_gens(
		face_df, 
		face_dir, 
		"gender", 
		"binary"
	)

	# Base CNN contains contolutional / pooling / FC layers.
	# Now we just add 1-node sigmoid layer for binary classification.
	gender_checkpt_path = os.path.join(out_dir, "gender.h5")
	gender_model = get_base_cnn()
	gender_model.add(Dropout(0.5))
	gender_model.add(Dense(1, activation='sigmoid'))
	gender_model.compile(
		optimizer='adam', 
		loss='binary_crossentropy', 
		metrics=['accuracy']
	)
	gender_loss, gender_acc = train_model(
		gender_model, 
		gender_train_gen, 
		gender_val_gen,
		gender_checkpt_path
	)

	print(f"Gender model summary:")
	gender_model.summary()
	print(f"Achieved {gender_loss} loss and {gender_acc} accuracy on 10%"
		f"validation set for gender model.")

	(age_train_gen, age_val_gen) = get_img_gens(
		face_df, 
		face_dir, 
		"age",
		"categorical"
	)
	age_checkpt_path = os.path.join(out_dir, "age.h5")
	age_model = get_base_cnn()
	age_model.add(Dense(16, activation='relu'))
	age_model.add(Dense(4, activation='softmax'))
	age_model.compile(
		optimizer='adam', 
		loss='categorical_crossentropy', 
		metrics=['accuracy']
	)
	age_loss, age_acc = train_model(
		age_model, 
		age_train_gen, 
		age_val_gen, 
		age_checkpt_path
	)

	print(f"Age model summary:")
	age_model.summary()
	print(f"Achieved {age_loss} loss and {age_acc} accuracy on 10%"
		f"validation set for age model.")
			
	

def train_text_model(in_dir, out_dir):
	pass
			
def train_like_model(in_dir, out_dir):
	pass

def train_models(in_dir, out_dir, modality):
	if modality == "image":
		train_image_models(in_dir, out_dir)
	elif modality == "text":
		train_text_model(in_dir, out_dir)
	elif modality == "like":
		train_like_model(in_dir, out_dir)
	
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
		train_models(args.input_dir, args.output_dir, args.train)

	else:
		print(f"Classifying")
		classify(args.input_dir, args.output_dir)
