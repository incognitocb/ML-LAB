import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

playtennis=pd.read_csv("Z:/SEM6/ML/datasets/playtennis.csv")
print("given dataset is: \n",playtennis,"\n")

le=LabelEncoder()
playtennis['outlook']=le.fit_transform(playtennis['outlook'])
playtennis['Temperature']=le.fit_transform(playtennis['Temperature'])
playtennis['Humidity']=le.fit_transform(playtennis['Humidity'])
playtennis['Wind']=le.fit_transform(playtennis['Wind'])
playtennis['PlayTennis']=le.fit_transform(playtennis['PlayTennis'])
print("the encoding dataset is \n",playtennis)

x=playtennis.drop(['PlayTennis'], axis=1)
y=playtennis['PlayTennis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
print("\n x_train : \n",x_train)
print("\n y_train : \n",y_train)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy= accuracy_score(y_test,y_pred)
print("\n Accuracy of Naive Bayes classifier:",accuracy)