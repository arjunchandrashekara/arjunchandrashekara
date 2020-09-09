import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing as p
import matplotlib.pyplot as plt
from sklearn import tree

data=pd.read_csv("E://excelr data//assignments//Decision Tree//Company_Data.csv")
data.columns
data.shape
data.dropna()
data.shape
np.mean(data.Sales)
def sales(x):
    if(x<=7.5):
        return("bad")
    else:
        return("good")


data['Sales'],_=pd.factorize(data['Sales'].apply(sales))
data['ShelveLoc'],_=pd.factorize(data['ShelveLoc'])
data['Urban'],_=pd.factorize(data['Urban'])
data['US'],_=pd.factorize(data['US'])


x=data.drop('Sales',axis=1)
y=data['Sales']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33)

help(DecisionTreeClassifier)

classifier=DecisionTreeClassifier(max_depth=10,criterion = 'entropy')
classifier.fit(x_train,y_train)
prediction=classifier.predict(x_test)
confusion_matrix(y_test,prediction)
pp=classifier.predict(x_train)

#tree
tree.plot_tree(classifier,filled=True)

#test accuracy
np.mean(y_test==prediction)

#train accuracy
np.mean(y_train==pp)

#bagging
from sklearn.ensemble import BaggingClassifier

result=[]
for i in range(1,23):
    classifier_with_bagging=BaggingClassifier(DecisionTreeClassifier(max_depth=10,criterion = 'entropy'),n_estimators=i,max_samples=200,random_state=20)
    classifier_with_bagging.fit(x_train,y_train)
    result.append(classifier_with_bagging.score(x_test,y_test))
result=pd.DataFrame(result)
#boosting
from sklearn.ensemble import AdaBoostClassifier

adamodel=AdaBoostClassifier(n_estimators=50)
model=adamodel.fit(x_train,y_train)
y_pred=model.predict(x_test)
np.mean(y_pred==y_test)
ada_result=[]
for i in range(1,50):
    abcd=AdaBoostClassifier(n_estimators=i,base_estimator=classifier,learning_rate=1)
    abcd.fit(x_train,y_train)
    y1_pred=abcd.predict(x_test)
    ada_result.append(np.mean(y1_pred==y_test))

ada_result=pd.DataFrame(ada_result)
    

