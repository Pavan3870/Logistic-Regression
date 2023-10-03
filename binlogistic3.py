import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import metrics

#loading data set
df = pd.read_csv("Social_Network_Ads.csv")
print(df.head())

# taking data to X and Y
X = df.iloc[:,2:4]
y = df.iloc[:,4]
#print(X.head())
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# Fitting Multiclass Logistic Classification to the Training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)
print("Score is :",lr.score(X_test,y_test))

print("Predicted VAlues:",y_pred)
print("testing VAlues:",y_test)

# Making the Confusion Matrix

cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion MAtrix:",cm)

#Acuuracy of the model
#from sklearn import metrics
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred)*100)


