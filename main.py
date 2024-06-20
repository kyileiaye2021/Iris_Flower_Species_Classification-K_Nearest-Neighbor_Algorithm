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
print("==========================Iris Flower Dataset========================")
print(df.head())
print(len(df))
print()

#========================Preparing the dataset===========================
#Checking and replacing the cells that are 0 with the mean of the column data
zero_not_accepted = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df[zero_not_accepted] = df[zero_not_accepted].fillna(df[zero_not_accepted].mean)

#Splitting dataset
X = df.drop("species", axis = 'columns')
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Calculate k value
k = int(np.sqrt(len(X_train_scaled))) #using square root heuristic
print("The number of neighbors: ", k)

#--------------------Fitting the model---------------------
#Define the KNN classifier model
classifier = KNeighborsClassifier(n_neighbors=k-1, p=2, metric="euclidean")
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
print()

#-------------------Evaluating the model with performance metrics-------------------
print("===============Evaluating the KNN Model================")
#Confusion Matrix
#A confusion matrix with no misclassifications would indicate that the model is performing perfectly on the given dataset, with 100% accuracy for all classes. 
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#f-1 score
# For macro-average F1-score
print("F1-score (macro):", f1_score(y_test, y_pred, average='macro'))

# For micro-average F1-score
print("F1-score (micro):", f1_score(y_test, y_pred, average='micro'))

# For weighted-average F1-score
print("F1-score (weighted):", f1_score(y_test, y_pred, average='weighted'))

#accuracy score
accuracy_score_of_model = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy_score_of_model)
print()

#===========================Predicting User Input=======================
print("==============Predicting Species of Input Iris Flower=============")
sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
user_input_pred = classifier.predict(user_input)
print(f"The species of the given iris flower is predicted to be {user_input_pred[0]}!") #return index 0 var because predict() returns numpy array
print()