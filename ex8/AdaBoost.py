from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
cancer = datasets.load_breast_cancer()
print("Features:", cancer.feature_names)
print("Labels:", cancer.target_names)
print(cancer.data.shape)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("##############################################################")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:",metrics.mean_squared_error(y_test, y_pred))
print("F-Measure:",metrics.recall_score(y_test, y_pred))
print("##############################################################")

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1

adbclassifier = AdaBoostClassifier(base_estimator=dtree,n_estimators=100,learning_rate=0.01,random_state=1)
model = adbclassifier.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))