import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp
plt.rcParams["figure.figsize"]=(10,6)

pip install scipy

data=pd.read_csv("Z:/SEM6/ML/Datasets/studentmarks.csv")
data.head()

plt.scatter(data['percentage'],data['Result'])
plt.show()
x=data['percentage']
y=data['Result']

X_train,X_test,y_train,y_test=train_test_split(data['percentage'],data['Result'],t
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(X_train.values.reshape(-1,1),y_train.values)
y_pred_sk=lr_model.predict(X_test.values.reshape(-1,1))
y_pred_sk

from scipy.special import expit
X_test=np.linspace(20,100,100)
y_test=X_test*lr_model.coef_+lr_model.intercept_
sigmoid=expit(y_test)
plt.scatter(data['percentage'],data['Result'])
plt.plot(X_test,sigmoid.ravel(),c='green',label='Logistic fit ')

