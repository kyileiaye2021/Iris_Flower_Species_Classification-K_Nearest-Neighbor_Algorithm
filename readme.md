# Predicting Species of Iris Flower with K Nearest Neighbor Algorithm

## K Nearest Neighbor Algorithm can be used for classifying objects and regression. But, in this project, it is used for classification.

### Concepts of K Nearest Neighbor in classification tasks
1. Set the value of k (num of neighbors)
2. Find the distance between the target datapoint and all other existing datapoint in the dataset
3. Arrange the distance in ascending order
4. Find the nearest distances in the range of k
5. Assign the target datapoint in the class based on the majority of vote 

### Steps of building K Nearest Classifier 
1. Dataset
   1. Import necessary libraries (pandas, numpy, sklearn) | confusion_matrix, f1_score, and accuracy_score are also imported from sklearn module for evaluating the model
   2. Load the dataset
2. Preparing dataset
   1. Splitting dataset
   2. Feature scaling
   3. Calculating k value (using square root heuristic)
      1. Calculate the total num of data in training set
      2. Take the square root of that total num
      3. If the res is even, make it odd by decrementing by 1
3. Fitting the model
   1. The euclidean metric with p=2 val is used
   2. Predict the X_test data
4. Evaluating the model
   1. Confusion Matrix
   2. F1-score
   3. Accuracy Score
5. Predicting the species of Input Iris Flower
   
### Results of the project
The accuracy of the K Nearest Neighbor (KNN) classifier model in this project is 100% (probably because of small dataset) as there is no misclassification in confusion matrix and the accuracy score is 1.0. Despite the 100% accuracy, the model can still be optimized by calculating k value with more advanced method such as **Cross Validation** instead of **Square Root Heuristic**.

### References
1. https://www.freecodecamp.org/news/k-nearest-neighbors-algorithm-classifiers-and-model-example/
2. https://www.youtube.com/watch?v=4HKqjENq9OU
