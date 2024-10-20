# Linear Regression with SVD for openness
# date: 10/12/2024
# name: Terri Bell
# description: Cross Validation tests, then training and pickling a model

import random
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD


# Reading the data into a dataframe and selecting the columns we need
df = pd.read_table("adjusted_relation_all_values.tsv")
data_Facebook = df.loc[:,['transcript', 'con']]

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

    # Vector count the data  in the likeid transcripts
    count_vect = CountVectorizer(max_df = .1263, min_df = 2)
    X_train = count_vect.fit_transform(data_train['transcript'])
    y_train = data_train['con']
    X_test = count_vect.transform(data_test['transcript'])
    y_test = data_test['con']

    # Reduce dimensionality via SVD
    svd = TruncatedSVD(n_components=100)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    # Train Linear Regression and make prediction
    linreg = LinearRegression()
    linreg.fit(X_train_svd, y_train)
    y_predicted = linreg.predict(X_test_svd)

    # Reporting on classification performance
    rmse = root_mean_squared_error(y_test, y_predicted)
    total += rmse
    print("Error: %.2f" % rmse)


average_error = total / 10.0
print("Average Error")
print(average_error)



'''

# Training a Linear Regression model using ALL training data,
# CountVectorizer excluding sites with more than 12.63% likes and less than 2 likes
# Using SVD to extract the highest 100 components
count_vect = CountVectorizer(max_df = 0.1263, min_df = 2)
X_train = count_vect.fit_transform(data_Facebook['transcript'])
y_train = data_Facebook['con']
svd = TruncatedSVD(n_components=100)
X_train_svd = svd.fit_transform(X_train)
linreg = LinearRegression()
linreg.fit(X_train_svd, y_train)

model_path = os.path.join('models', 'lin_reg_svd_con.pkl')
with open(model_path, 'wb') as fout:
    pickle.dump((linreg, count_vect, svd), fout)
