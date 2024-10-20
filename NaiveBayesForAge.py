# First Run, Facebook Likes to predict Gender
# date: 10/12/2024
# name: Terri Bell
# description: Naive Bayes model for gender recognition of Facebook users

import random
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Reading the data into a dataframe and selecting the columns we need
df = pd.read_table("adjusted_relation_age.tsv")
#print(df.shape)


data_Facebook = df.loc[:,['transcript', 'age']]
#print(data_Facebook)


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
    #train_Ids = all_Ids[n*(i+1):]
    data_test = data_Facebook.loc[test_Ids, :]
    data_train = data_Facebook.loc[train_Ids, :]

    # Training a Naive Bayes model
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data_train['transcript'])
    y_train = data_train['age']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    X_test = count_vect.transform(data_test['transcript'])
    y_test = data_test['age']
    y_predicted = clf.predict(X_test)

    # Reporting on classification performance
    total += accuracy_score(y_test, y_predicted)
    print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
    classes = [0, 1, 2, 3]
    cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)

average_accuracy = total / 10.0
print("Average Accuracy")
print(average_accuracy)
'''

# Training a Naive Bayes model using ALL training data
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_Facebook['transcript'])
y_train = data_Facebook['age']
clf = MultinomialNB()
clf.fit(X_train, y_train)
model_path = os.path.join('results', 'naive_bayes_model_age.pkl')
with open(model_path, 'wb') as fout:
    pickle.dump((clf, count_vect), fout)

'''