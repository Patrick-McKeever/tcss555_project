# Preprocessing 
# date: 10/17/2024
# name: Terri Bell
# description: read in a .csv file consisting of rows: user_id likeid 
# output a .csv file consisting of rows: user_id gender age 5 personality traits transcirpt num_likes
# where all likeid's for a given user are combined into a single string and num_likes is the number of likeid's

import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import argparse
import os
import sys
import csv

# Read the relation.csv and collect all like_ids on one row in new output file
def generate_transcript():
    count = 0 
    output_path = os.path.join('results', 'adjusted_relation_all_values.tsv')
    with open(output_path, "w") as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_header = ['', 'userid', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu', 'transcript', 'num_likes']
        tsv_writer.writerow(tsv_header)
        with open ("profile.csv", mode = 'r') as profile_file:
            profile_reader = pd.read_csv(profile_file)
            with open("relation.csv", mode='r') as relation_file:
                relation_reader = csv.DictReader(relation_file)
                current_user = ""
                new_user = ""
                current_transcript = ""
                num = 0
                for row in relation_reader:
                    new_user = row['userid']
                    #print(new_user)
                    if (new_user == current_user):
                        # if the row is for the same user, add the like_id to the transcript
                        current_transcript = current_transcript + " " + row['like_id'].rstrip()
                        # increase like count by 1
                        num += 1
                    elif (current_user == ""):
                        #first user
                        current_transcript = row['like_id'].rstrip()
                        current_user = new_user
                        num += 1
                    else:
                        # row corresponds to a new user, print output and start a new transcript
                        profile = profile_reader[profile_reader['userid'] == current_user]
                       
                        age = profile['age'].values[0]
                        age_class = 0
                        if age <= 24:
                            age_class = 0
                        elif age < 35:
                            age_class = 1
                        elif age < 50:
                            age_class = 2
                        else:
                            age_class = 3
                        #print(age)
                        gender = profile['gender'].values[0]
                        ope = profile['ope'].values[0]
                        con = profile['con'].values[0]
                        ext = profile['ext'].values[0]
                        agr = profile['agr'].values[0]
                        neu = profile['neu'].values[0]
                        out_row = [count, current_user, age_class, gender, ope, con, ext, agr, neu, current_transcript, num]
                        tsv_writer.writerow(out_row)
                        # set initial values for new user to be current user
                        current_user = new_user
                        current_transcript = row['like_id']
                        count += 1
                        num = 1
                # write the last user
                profile = profile_reader[profile_reader['userid'] == current_user]
                age = profile['age'].values[0]
                age_class = 0
                if age <= 24:
                    age_class = 0
                elif age < 35:
                    age_class = 1
                elif age < 50:
                    age_class = 2
                else:
                    age_class = 3
                gender = profile['gender'].values[0]
                ope = profile['ope'].values[0]
                con = profile['con'].values[0]
                ext = profile['ext'].values[0]
                agr = profile['agr'].values[0]
                neu = profile['neu'].values[0]
                out_row = [count, current_user, age_class, gender, ope, con, ext, agr, neu, current_transcript, num]
                tsv_writer.writerow(out_row)
                

generate_transcript()                

