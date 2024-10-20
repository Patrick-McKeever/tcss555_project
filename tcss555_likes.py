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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD

def check_valid_directory(path):
	if not os.path.isdir(path):
		raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
	return path

				
def generate_one_output(out_dir, current_user, age_class, gender, ope, neu, ext, agr, con):
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
				extrovert=\"{ext}\"
				neurotic=\"{neu}\"
				agreeable=\"{agr}\"
				conscientious=\"{con}\"
				open=\"{ope}\"
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
	gender_model_path = os.path.join(model_path, "models", "log_reg_gender.pkl")
	age_model_path = os.path.join(model_path, "models", "log_reg_age.pkl")
	ope_model_path = os.path.join(model_path, "models", "lin_reg_svd_ope.pkl")
	neu_model_path = os.path.join(model_path, "models", "lin_reg_svd_neu.pkl")
	ext_model_path = os.path.join(model_path, "models", "lin_reg_svd_ext.pkl")
	agr_model_path = os.path.join(model_path, "models", "lin_reg_svd_agr.pkl")
	con_model_path = os.path.join(model_path, "models", "lin_reg_svd_con.pkl")
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
			with open(ope_model_path, 'rb') as ope_model_file:
				with open(neu_model_path, 'rb') as neu_model_file:
					with open(ext_model_path, 'rb') as ext_model_file:
						with open(agr_model_path, 'rb') as agr_model_file:
							with open(con_model_path, 'rb') as con_model_file:
								gender_reg, gender_count = pickle.load(gender_model_file)
								age_reg, age_count = pickle.load(age_model_file)
								ope_reg, ope_count, ope_svd = pickle.load(ope_model_file)
								neu_reg, neu_count, neu_svd = pickle.load(neu_model_file)
								ext_reg, ext_count, ext_svd = pickle.load(ext_model_file)
								agr_reg, agr_count, agr_svd = pickle.load(agr_model_file)
								con_reg, con_count, con_svd = pickle.load(con_model_file)
							gender_X = gender_count.transform(test_data['transcript'])
							gender_y_predicted = gender_reg.predict(gender_X)
							age_X = age_count.transform(test_data['transcript'])
							age_y_predicted = age_reg.predict(age_X)
							ope_X = ope_count.transform(test_data['transcript'])
							ope_X_svd = ope_svd.transform(ope_X)
							ope_y_predicted = ope_reg.predict(ope_X_svd)
							neu_X = neu_count.transform(test_data['transcript'])
							neu_X_svd = neu_svd.transform(neu_X)
							neu_y_predicted = neu_reg.predict(neu_X_svd)
							ext_X = ext_count.transform(test_data['transcript'])
							ext_X_svd = ext_svd.transform(ext_X)
							ext_y_predicted = ext_reg.predict(ext_X_svd)
							agr_X = agr_count.transform(test_data['transcript'])
							agr_X_svd = agr_svd.transform(agr_X)
							agr_y_predicted = agr_reg.predict(agr_X_svd)
							con_X = con_count.transform(test_data['transcript'])
							con_X_svd = con_svd.transform(con_X)
							con_y_predicted = con_reg.predict(con_X_svd)
							for i in range(len(test_data)):
								userid = input_data.iloc[i].loc['userid']
								age = age_y_predicted[i]
								gender = gender_y_predicted[i]
								ope = ope_y_predicted[i]
								neu = neu_y_predicted[i]
								ext = ext_y_predicted[i]
								agr = agr_y_predicted[i]
								con = con_y_predicted[i]
								generate_one_output(out_dir, userid, age, gender, ope, neu, ext, agr, con)

			
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Check input and output directory paths.')

	parser.add_argument('-i', '--input_dir', required=True, type=check_valid_directory,
						help='Path to the input directory')
	parser.add_argument('-o', '--output_dir', required=True, type=check_valid_directory,
						help='Path to the output directory')

	args = parser.parse_args()
	generate_outputs(args.input_dir, args.output_dir)
