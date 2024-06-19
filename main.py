#IRIS FLOWER SPECIES CLASSIFICATION - K Nearest Neighbor Algorithm

'''
K Nearest Neighbor Algorithm (can be used for classifying objects and regression)
In this project, K Nearest Neighbor is used for classification
1. K value is set (usually odd num | sqrt of the num of training dataset)
2. Distance between datapoint entry and all other existing datapoints are calculated and sorted in ascending order
3. Find K nearest neighbor to the datapoint entry 
4. Classify the datapoint entry based on the majority vote of K nearest neighbors
'''


#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #for separating training and testing data
from sklearn.preprocessing import StandardScaler #for feature scaling
from sklearn.neighbors import KNeighborsClassifier #for building k nearest neighbor classifier model
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score  #for testing the model

#loading the dataset
data = pd.read_csv("IRIS.csv")
df = pd.DataFrame(data)
print(df.head())
print(len(df))

#========================Preparing the dataset===========================
#Checking and replacing the cells that are 0 with the mean of the column data
zero_not_accepted = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[zero_not_accepted] = df[zero_not_accepted].fillna(df[zero_not_accepted].mean)

#Splitting dataset
X = df.drop("species")
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--------------------Fitting the model---------------------
#Define the KNN classifier model
