# !/usr/bin/python3

import argparse
import os
import io
import sys
import csv
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def check_valid_directory(path):
	if not os.path.isdir(path):
		raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
	return path

				
def generate_one_output(out_dir, current_user, gender, age_class):
	age = "xx-24"
	if age_class == 1:
		age = "25-34"
	elif age_class == 2:
		age = "35-49"
	elif age_class == 3:
		age = "50-xx"
	gender_string = "female"
	if gender == 0.0:
		gender_string = "male"
	outstr = f"""<user
				id=\"{current_user}\"
				age_group=\"{age}\"
				gender=\"{gender_string}\"
				extrovert=\"3.486857895\"
				neurotic=\"2.732424\"
				agreeable=\"3.583904211\"
				conscientious=\"3.445616842\"
				open=\"3.908690526\"
			/>"""
	outpath = os.path.join(out_dir, current_user + '.xml')
	with open(outpath, 'w+') as outf:
		outf.write(outstr)
	

def preprocessing(in_dir):
	count = 0
	tsv_data = io.StringIO()
	tsv_writer = csv.writer(tsv_data, delimiter='\t')
	tsv_header = ['', 'userid', 'transcript']
	tsv_writer.writerow(tsv_header)
	relation_path = os.path.join(in_dir, "relation", "relation.csv")
	with open(relation_path, mode='r') as relation_file:
		relation_reader = csv.DictReader(relation_file)
		current_user = ""
		new_user = ""
		current_transcript = ""
		for row in relation_reader:
			new_user = row['userid']
			if (new_user == current_user):
				# if the row is for the same user, add the like_id to the transcript
				current_transcript = current_transcript + " " + row['like_id'].rstrip()
			elif (current_user == ""):
				#first user
				current_transcript = row['like_id'].rstrip()
				current_user = new_user
			else:
				# row corresponds to a new user, print output and start a new transcript
				out_row = [count, current_user, current_transcript]
				tsv_writer.writerow(out_row)
				current_user = new_user
				current_transcript = row['like_id']
				count += 1
			# write the last user
			out_row = [count, current_user, current_transcript]
			tsv_writer.writerow(out_row)
	return tsv_data.getvalue()
	

def generate_outputs(in_dir, out_dir):
	model_path = os.getcwd()
	gender_model_path = os.path.join(model_path, "models", "naive_bayes_model.pkl")
	age_model_path = os.path.join(model_path,"models", "naive_bayes_model_age.pkl")
	relation_path = os.path.join(in_dir, "relation", "relation.csv")
	if not os.path.isfile(relation_path):
		print(f"{relation_path} not found, exiting")
		exit(1)
	processed_data = preprocessing(in_dir)
	df = pd.read_csv(io.StringIO(processed_data), sep='\t')
	input_data = df.loc[:, ['userid', 'transcript']]
	test_data = df.loc[:,['transcript']]
	with open(gender_model_path, 'rb') as gender_model_file:
		with open(age_model_path, 'rb') as age_model_file:
			gender_clf, gender_count_vect = pickle.load(gender_model_file)
			age_clf, age_count_vect = pickle.load(age_model_file)
			gender_X_predict = gender_count_vect.transform(test_data['transcript'])
			gender_Y_predicted = gender_clf.predict(gender_X_predict)
			age_X_predict = age_count_vect.transform(test_data['transcript'])
			age_Y_predicted = age_clf.predict(age_X_predict)
		for i in range(len(test_data)):
			generate_one_output(out_dir, input_data.iloc[i].loc['userid'], gender_Y_predicted[i], age_Y_predicted[i])


			
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Check input and output directory paths.')

	parser.add_argument('-i', '--input_dir', required=True, type=check_valid_directory,
						help='Path to the input directory')
	parser.add_argument('-o', '--output_dir', required=True, type=check_valid_directory,
						help='Path to the output directory')

	args = parser.parse_args()
	generate_outputs(args.input_dir, args.output_dir)
