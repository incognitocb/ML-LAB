import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('z:\ML\playtennis.csv')

data

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()

data['outlook']=Le.fit_transform(data['outlook'])
data['Temperature']=Le.fit_transform(data['Temperature'])
data['Humidity']=Le.fit_transform(data['Humidity'])
data['Wind']=Le.fit_transform(data['Wind'])
data['PlayTennis']=Le.fit_transform(data['PlayTennis'])
data

x=data.drop(['PlayTennis'],axis=1)
y=data['PlayTennis']
from sklearn import tree
import matplotlib
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(x,y)
tree.plot_tree(clf)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris=load_iris()
X=iris.data[:,2:] #petal length and width
Y=iris.target
tree_clf=DecisionTreeClassifier(max_depth=2,random_state=42)
tree_clf.fit(X,Y)
from sklearn.tree import export_graphviz
export_graphviz(
	tree_clf,
	out_file="iris_tree.dot",
	feature_names=iris.feature_names[2:],
	class_names=iris.target_names,
	rounded=True,
	filled=True)
from graphviz import Source
Source.from_file('iris_tree.dot')

%pip install graphviz

import graphviz
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
graph

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
y_pred=clf.predict(x_test)
cf_matrix=confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cf_matrix)
print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)

import seaborn as sns
ax=sns.heatmap(cf_matrix,annot=True,cmap='Blues')
ax.set_title('confusion Matrix with labels\n\n');
ax.set_xlabel('\npredicted values')
ax.set_ylabel('Actual values');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

