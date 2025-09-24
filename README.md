#   EX NO 7 : SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.enerate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Vignesh Raaj S
RegisterNumber: 212223230239
*/
PROGRAM

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:
<img width="830" height="297" alt="image" src="https://github.com/user-attachments/assets/be3bff50-6f6d-457e-b50a-5b355d257d15" />

<img width="502" height="104" alt="image" src="https://github.com/user-attachments/assets/4e5d1baf-1fb4-47f8-b00b-34ccf550ed7b" />

<img width="699" height="34" alt="image" src="https://github.com/user-attachments/assets/17357759-6a07-4ceb-bf6b-010abc0680ad" />

<img width="553" height="120" alt="image" src="https://github.com/user-attachments/assets/a99cdcb0-5059-4e4e-a42a-08425fffaae7" />

<img width="640" height="251" alt="image" src="https://github.com/user-attachments/assets/7125d690-df4f-4c09-8be2-b9987fdc04ce" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
