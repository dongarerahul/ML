import numpy as np
import matplotlib.pyplot as plt # plot graphs
import pandas as pd #import datasets

dataset = pd.read_csv('Data.csv')

#create matrix of independent features / variables (country, age, salary)
X = dataset.iloc[:, :-1].values

#create dependent value matrix
y = dataset.iloc[:, 3].values

#taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
