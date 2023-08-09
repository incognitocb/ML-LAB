import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Z:\SEM6\ML\Datasets\salary_data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_st
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('Salary VS Experience (Training set)')
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('Salary VS Experience (Test set)')
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()

y_pred = regressor.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgd=SGDRegressor(max_iter=6000,penalty=None,eta0=0.1,alpha=0.01)
X_train=X_train.reshape(-1,1)
sgd.fit(X_train,y_train)
X_test=X_test.reshape(-1,1)
y_pred=sgd.predict(X_test)
plt.plot(X_test,y_pred,marker='o',
	color='blue',markerfacecolor='red',
	markersize=10,linestyle='dashed')
plt.scatter(X,y,marker='o',color='red')
plt.xlabel("yexp")
plt.ylabel("slaary")
plt.title("Stochastic Gradient descent")
plt.show()