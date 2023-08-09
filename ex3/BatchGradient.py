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

# Batch Gradient Descent

y_pred = regressor.predict(X_test)
plt.plot(X,y,'bo')

plt.plot(X,y,'bo')
m=0
b=0
Y=lambda x:m*x+b
def plot_line(Y,data_points):
	x_values=[i for i in range(int(min(data_points))-1,int(max(data_points))+2)]
	y_values=[Y(x) for x in x_values]
	plt.plot(x_values,y_values,'r')
plot_line(Y,X)
#After executing the plot_line function
plt.plot(X,y,"bo")
learn=.01
def summation(Y,X,y):
	total1=0;
	total2=0;
	for i in range(1,len(X)):
		total1+=Y(X[i])-y[i]
		total2+=(Y(X[i])-y[i])+X[i]
	return total1/len(X),total2/len(y)
for i in range(50):
	s1,s2=summation(Y,X,y)
	m=m-learn*s2
	b=b-learn*s1
	plot_line(Y,X)
	plt.plot(X,y,"bo")