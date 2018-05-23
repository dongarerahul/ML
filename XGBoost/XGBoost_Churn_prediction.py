#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# data preprocessing
# Encode country, gender columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encode Geography
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])

# Encode Gender
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Dataset splitting
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle    

X, y = shuffle(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Classifer training
from xgboost import XGBClassifier
classifier = XGBClassifier()

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
 
classifier.fit(X_train, y_train)
y_pred = cross_val_predict(classifier, X_test, y_test, cv=10)
result_metrics = confusion_matrix(y_test, y_pred)
print(result_metrics)

# confusion matrix
#1521	74
#197	208
