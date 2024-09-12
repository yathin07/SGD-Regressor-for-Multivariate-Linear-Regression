# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function 
5.Parameter Update Rule 
6.Iterative Training 
7.Model Evaluation 
8.End

## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Yathin Reddy T
RegisterNumber: 212223100062

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```

## Output:
![image](https://github.com/user-attachments/assets/15c84b51-a0fc-4485-a744-c6b54b213c07)

df.info()df.info()

![image](https://github.com/user-attachments/assets/39b84a4b-9713-4278-b94a-12a9086cb09f)


X=df.drop(columns=['AveOccup','target'])
X.info()

![image](https://github.com/user-attachments/assets/d65880ed-bf30-4603-8c4f-b613df13e2cc)

Y=df[['AveOccup','target']]
Y.info()

![image](https://github.com/user-attachments/assets/9b311313-538d-4874-9f01-28916767f820)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()

![image](https://github.com/user-attachments/assets/46b12852-59b4-498e-9200-e6ec3a603f81)

scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

![image](https://github.com/user-attachments/assets/3a10a3f5-8143-4bed-9657-eaea71120622)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])

![image](https://github.com/user-attachments/assets/7e3a6eb8-9bac-4e15-9c4e-967d8a183cfd)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
