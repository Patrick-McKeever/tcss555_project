# Logistic Regression for Facebook likes and gender
# date: 10/12/2024
# name: Terri Bell
# description: Logistic Regression model for gender recognition of Facebook users

import random
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# Reading the data into a dataframe and selecting the columns we need
df = pd.read_table("adjusted_relation_all_values.tsv")

data_Facebook = df.loc[:,['transcript', 'gender']]

'''
#Performing 10-fold validation 
# Splitting the data into 8550 training instances and 950 test instances 10 times
n = 950
all_Ids = np.arange(len(data_Facebook))
print(len(data_Facebook)) 
random.shuffle(all_Ids)
total = 0.0

for i in range(10):
    print(i)
    test_Ids = all_Ids[i*n:n*(i+1)]
    train_Ids = np.delete(all_Ids, slice(i*n, n*(i+1)))

    data_test = data_Facebook.loc[test_Ids, :]
    data_train = data_Facebook.loc[train_Ids, :]

    # Training a Logistic Regression Model
    count_vect = CountVectorizer(max_df=0.12, min_df=2)
    X_train = count_vect.fit_transform(data_train['transcript'])
    y_train = data_train['gender']
    logreg = LogisticRegression(C=100.0, penalty='l2', solver='sag')
    logreg.fit(X_train, y_train)

    # Testing the Logistic Regression model
    X_test = count_vect.transform(data_test['transcript'])
    y_test = data_test['gender']
    y_predicted = logreg.predict(X_test)

    # Reporting on classification performance
    total += accuracy_score(y_test, y_predicted)
    print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
    classes = [0, 1]
    cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)

average_accuracy = total / 10.0
print("Average Accuracy")
print(average_accuracy)
'''

# Training a Logistic Regression model using ALL training data
count_vect = CountVectorizer(max_df = 0.1263, min_df = 2)
X_train = count_vect.fit_transform(data_Facebook['transcript'])
y_train = data_Facebook['gender']
logreg = LogisticRegression(C=10.0, penalty='l2', solver='sag')
logreg.fit(X_train, y_train)
model_path = os.path.join('models', 'log_reg_gender.pkl')
with open(model_path, 'wb') as fout:
    pickle.dump((logreg, count_vect), fout)