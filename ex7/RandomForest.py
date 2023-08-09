import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score

iris=load_iris()
x=iris.data
y=iris.target
clf=RandomForestClassifier(n_estimators=5)
clf.fit(x,y)

for i,tree_in_forest in enumerate(clf.estimators_):
	fig=plt.figure(figsize=(10,10))
	_=tree.plot_tree(tree_in_forest,
					feature_names=iris.feature_names,
					class_names=iris.target_names,
					filled=True)
	fig.savefig(f'tree_{i}.png')
	y_pred=tree_in_forest.predict(x)
	accuracy=np.mean(y_pred==y)*100
	print(f"Accuracy for Tree {i}:{accuracy:.2f}%")
y_pred_rf=clf.predict(x)
accuracy_rf=accuracy_score(y,y_pred_rf)*100
print(f"Overall Random Forest Accuracy:{accuracy_rf:.2f}% ")

from sklearn.model_selection import train_test_split

file_path="Z:/6th/ML/datasets/iris2.csv"
data=pd.read_csv(file_path)
x=data.drop('class',axis=1)
y=data['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1
clf=RandomForestClassifier(n_estimators=5)
clf.fit(x_train,y_train)

for i,tree_in_forest in enumerate(clf.estimators_):
	tree.plot_tree(tree_in_forest,feature_names=data.columns[:-1],class_names=d
	plt.show()
	y_pred=clf.predict(x_test)
	accuracy=np.mean(y_pred==y_test)*100
	print(f"Accuracy for Tree {i}:{accuracy:.2f}%")
y_pred_rf=clf.predict(x_train)
accuracy_rf=accuracy_score(y_train,y_pred_rf)*100
print(f"Overall Random Forest Accuracy: {accuracy_rf:.2f}%")