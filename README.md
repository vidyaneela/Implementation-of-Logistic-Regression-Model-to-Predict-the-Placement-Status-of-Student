# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vidya Neela.M
RegisterNumber:  212221230120
*/
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Head:
![196478925-813e8a4e-b32a-4ef3-b88e-cac875b00d3b](https://user-images.githubusercontent.com/94169318/200601029-b3844ee9-c508-4951-ba0d-84e8ed941789.png)

### Predicted values::
![196478995-570d7c28-ec74-4a22-a986-b9268447b217](https://user-images.githubusercontent.com/94169318/200601113-f3e2b563-f944-48dd-9c2c-15812bfb3754.png)

### Accuracy:
![196479062-da41501a-7c2d-4702-9a25-44a43772eb02](https://user-images.githubusercontent.com/94169318/200601193-bacc3d7b-e9d9-42b8-969a-be5c3007dc3a.png)

### Confusion Matrix:
![196479116-ac364f28-141d-43fe-8f5f-ed9b740fbf41](https://user-images.githubusercontent.com/94169318/200601283-146bc567-81dd-474e-92a6-de911309fd49.png)

### Classification Report:
![196479205-7da2b9e1-5a0f-40bd-9b3f-eb8fcc6e94e8](https://user-images.githubusercontent.com/94169318/200601368-67a357d9-aa2e-4d37-ba67-98bffd477e16.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
