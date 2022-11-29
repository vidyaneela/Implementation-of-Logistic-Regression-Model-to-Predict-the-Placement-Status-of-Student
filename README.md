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
### Original data(first five columns):
![o1](https://user-images.githubusercontent.com/94169318/204547952-ec39c3a4-d040-4917-8eb5-958903014530.png)

### Data after dropping unwanted columns(first five):
![o2](https://user-images.githubusercontent.com/94169318/204548023-d38dcb5a-d3d3-4ebe-a834-ebd9d0b39330.png)

### Checking the presence of null values:
![o3](https://user-images.githubusercontent.com/94169318/204548087-b72930cb-fb18-475b-bd7c-03ca765b2e6b.png)

### Checking the presence of duplicated values:
![o4](https://user-images.githubusercontent.com/94169318/204548149-e645b399-f4f8-43e5-afa1-724143469fff.png)

### Data after Encoding:
![o5](https://user-images.githubusercontent.com/94169318/204548255-634deb97-d4e8-4088-bab5-7f7b6d9d07ba.png)

### X Data:
![o6](https://user-images.githubusercontent.com/94169318/204548344-264be899-9e38-4350-9d18-3e9c2b6a9e28.png)

### Y Data:
![o7](https://user-images.githubusercontent.com/94169318/204548495-be450f16-a693-4aaa-a328-d78a8c6c5c02.png)

### Predicted Values:
![o8](https://user-images.githubusercontent.com/94169318/204548574-cb857a92-e3ae-431d-8db8-ce9de442130a.png)

### Accuracy Score:
![o9](https://user-images.githubusercontent.com/94169318/204548661-93fe7cf3-7c03-4b2d-8854-2042d460dc04.png)

### Confusion Matrix:
![o10](https://user-images.githubusercontent.com/94169318/204548748-bf5da3c6-9bd2-46b6-8b30-6c7354ff835b.png)

### Classification Report:
![o11](https://user-images.githubusercontent.com/94169318/204548825-0e83dd46-ef2b-427e-89d9-35966fb0e622.png)

### Predicting output from Regression Model:
![o12](https://user-images.githubusercontent.com/94169318/204548881-4ee2870c-fb88-4bc8-8a44-6c175b2e52b7.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
